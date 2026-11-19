"""Randomized A* evader routes.

A route chains shortest paths between random goals, is fitted with one cubic
B-spline and timed by path_motion.profiled_waypoints. The kinematic runtime
lives in task/target.py.
"""
from __future__ import annotations

import math

import torch

from target_interception.scenarios import path_motion
from target_interception.scenarios.generation_stats import count as count_generation
from target_interception.scenarios.generation_stats import time_stage
from target_interception.task import target

from .coverage import TARGET_SPEED_BIN_EDGES, CoverageRequest
from .grid import (
    grid_cell_xy,
    grid_cells_to_xy,
    grid_cells_xy,
    grid_free_cells,
    grid_shortest_path_tree,
    grid_tree_path,
    grid_tree_reachable_cells,
)


def evader_step_distances(env, speed: float) -> torch.Tensor:
    dt_steps = (env._path_waypoint_steps[1:] - env._path_waypoint_steps[:-1]).to(torch.float32)
    return dt_steps * env._step_dt * float(speed)


def grid_static_evader_waypoints(env, grid: dict, cell: tuple[int, int], z: float) -> torch.Tensor:
    waypoints = torch.zeros(env._path_waypoint_count, 3, device=env.device)
    waypoints[:, :2] = grid_cell_xy(grid, cell).view(1, 2)
    waypoints[:, 2] = float(z)
    return waypoints


def sample_grid_evader_waypoints(
    env,
    grid: dict,
    start_cell: tuple[int, int],
    z: float,
    *,
    step_dist: torch.Tensor,
    preferred_initial_direction: torch.Tensor | None = None,
    preferred_initial_min_alignment: float = 0.3,
    avoid_mask: torch.Tensor | None = None,
    avoid_distance: float = 0.0,
) -> torch.Tensor | None:
    """Chain A* segments between random goals, then fit and time one spline.

    Segments that start within avoid_distance of route length also treat
    avoid_mask (the interceptor spawn square) as occupied.
    """
    step_dist = torch.as_tensor(step_dist, device=env.device, dtype=torch.float32)
    required_distance = float(torch.sum(step_dist).item())
    if required_distance <= 0.0:
        return grid_static_evader_waypoints(env, grid, start_cell, z)
    # Smoothing shortens the polyline, so routes are built ~8% longer than
    # required; timing ignores the unused tail, while a short route is rejected.
    needed = required_distance * 1.08 + float(grid["cell_size"])
    goal_attempts = max(1, int(env.cfg.scenario.smooth_evader_goal_attempts))
    backtracks_left = max(0, int(env.cfg.scenario.smooth_evader_backtracks))

    occupied_early = grid["occupied"] if avoid_mask is None else grid["occupied"] | avoid_mask

    current = start_cell
    polyline = grid_cell_xy(grid, current).view(1, 2)
    distance = 0.0
    # (polyline, current) before each segment, so a stuck route can back up and
    # re-extend; more goals from the same dead end would not help.
    history: list[tuple[torch.Tensor, tuple[int, int]]] = []
    segment_budget = 24 + backtracks_left
    for _ in range(segment_budget):
        if distance >= needed:
            break
        # Sample eligible destinations, then validate the actual path junction.
        segment_id = len(history)
        occupied = occupied_early if distance < avoid_distance else grid["occupied"]
        initial_direction = preferred_initial_direction if segment_id == 0 else None
        candidates = grid_goal_candidates(
            env,
            grid,
            occupied,
            current,
            initial_direction=initial_direction,
            initial_min_alignment=preferred_initial_min_alignment,
            goal_min_override=0.35 if initial_direction is not None else None,
        )

        next_polyline = None
        incoming = junction_incoming(env, polyline) if segment_id > 0 else None
        if candidates is not None:
            search_occupied = occupied
            search_start = current
            if initial_direction is not None:
                search_start = preferred_grid_step(
                    occupied, current, initial_direction, preferred_initial_min_alignment
                )
                if search_start is None:
                    count_generation(env, "route.initial_motion")
                    return None
                # Force the first step instead of hoping a shortest-path tie picks
                # it. Blocking the old start also prevents an immediate U-turn.
                search_occupied = occupied.clone()
                search_occupied[current] = True
            with time_stage(env, "route_search"):
                tree = _route_tree(grid, search_occupied, search_start)
            reachable_candidates = grid_tree_reachable_cells(tree, candidates)
            order = torch.randperm(reachable_candidates.shape[0], device=env.device)
            for pick in order[:goal_attempts].tolist():
                count_generation(env, "route.proposals")
                goal = (
                    int(reachable_candidates[pick, 0]),
                    int(reachable_candidates[pick, 1]),
                )
                path = grid_tree_path(tree, search_start, goal)
                if path is not None and search_start != current:
                    path = [current] + path
                if path is None or len(path) <= 1:
                    count_generation(env, "route.unreachable")
                    continue
                path_xy = grid_cells_to_xy(env, grid, path)
                if initial_direction is not None:
                    actual = torch.nn.functional.normalize(path_xy[1] - path_xy[0], dim=0)
                    if float(torch.dot(actual, initial_direction).item()) < preferred_initial_min_alignment:
                        count_generation(env, "route.initial_motion")
                        continue
                if segment_id > 0:
                    with time_stage(env, "smoothing_connection"):
                        if not junction_aligned(env, polyline, incoming, path_xy):
                            continue
                next_polyline = torch.cat((polyline, path_xy[1:]), dim=0)
                break

        if next_polyline is None:
            if backtracks_left <= 0 or not history:
                count_generation(env, "route.unreachable")
                return None
            backtracks_left -= 1
            polyline, current = history.pop()
            distance = polyline_length_xy(polyline)
            continue
        history.append((polyline, current))
        polyline = next_polyline
        distance = polyline_length_xy(polyline)
        current = path[-1]

    # Snapshots end at the actual A* goals and naturally follow backtracking.
    junctions = [polyline_length_xy(prefix) for prefix, _ in history[1:]]
    with time_stage(env, "smoothing_connection"):
        return path_motion.spline_waypoints(env, grid, polyline, junctions, step_dist, z)


def junction_incoming(env, previous: torch.Tensor):
    """Incoming chord over the blend distance, shared by every candidate goal."""
    previous_s = path_motion.arc_lengths(previous)
    blend = min(float(env.cfg.scenario.smooth_evader_goal_blend_distance), 0.45 * float(previous_s[-1]))
    return previous_s, blend, _chord_in(previous, previous_s, blend)


def junction_aligned(env, previous: torch.Tensor, incoming, following: torch.Tensor) -> bool:
    """Reject a join whose direction turns back beyond the configured alignment.

    Incoming and outgoing directions are chords over the blend distance on each
    side of the join; the spline fit smooths the accepted corner.
    """
    previous_s, blend, direction_in = incoming
    following_s = path_motion.arc_lengths(following)
    if 0.45 * float(following_s[-1]) < blend:  # a short segment shortens both chords
        blend = 0.45 * float(following_s[-1])
        direction_in = _chord_in(previous, previous_s, blend)
    if blend <= 1e-6:
        count_generation(env, "route.junction_degenerate")
        return False
    exit = path_motion.interpolate(following_s, following, following_s.new_tensor([blend]))[0]
    direction_out = torch.nn.functional.normalize(exit - following[0], dim=0)
    if float(torch.dot(direction_in, direction_out)) < float(env.cfg.scenario.smooth_evader_goal_min_alignment):
        count_generation(env, "route.junction_alignment")
        return False
    return True


def _chord_in(previous: torch.Tensor, previous_s: torch.Tensor, blend: float) -> torch.Tensor:
    entry = path_motion.interpolate(previous_s, previous, previous_s[-1:] - blend)[0]
    return torch.nn.functional.normalize(previous[-1] - entry, dim=0)


def preferred_grid_step(
    occupied: torch.Tensor,
    start: tuple[int, int],
    direction: torch.Tensor,
    minimum_alignment: float,
) -> tuple[int, int] | None:
    """Best collision-free first grid step in the requested motion bucket."""
    row, col = start
    rows, cols = (int(value) for value in occupied.shape)
    candidates = []
    for dr, dc in (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ):
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or bool(occupied[nr, nc].item()):
            continue
        if dr != 0 and dc != 0 and bool((occupied[row, nc] | occupied[nr, col]).item()):
            continue
        step = direction.new_tensor((float(dc), float(dr)))
        alignment = float(torch.dot(torch.nn.functional.normalize(step, dim=0), direction).item())
        if alignment >= minimum_alignment:
            candidates.append((alignment, nr, nc))
    if not candidates:
        return None
    _, nr, nc = max(candidates)
    return nr, nc


def grid_goal_candidates(
    env,
    grid: dict,
    occupied: torch.Tensor,
    start_cell: tuple[int, int],
    *,
    initial_direction: torch.Tensor | None = None,
    initial_min_alignment: float = 0.3,
    goal_min_override: float | None = None,
) -> torch.Tensor | None:
    """Eligible free destinations ([K, 2] rows/cols).

    Apply distance and arena-inset bounds, with an optional preference for the
    requested initial motion. Continuations have no destination-bearing filter:
    their junction is validated on the extracted shortest path.
    """
    free = torch.nonzero(~occupied, as_tuple=False)
    if free.numel() == 0:
        return None

    start_xy = grid_cell_xy(grid, start_cell)
    goal_min = max(
        0.0,
        float(
            goal_min_override
            if goal_min_override is not None
            else env.cfg.scenario.evader_goal_min_distance
        ),
    )
    xy = grid_cells_xy(grid, free)
    distance = torch.linalg.vector_norm(xy - start_xy.view(1, 2), dim=-1)
    valid = distance >= goal_min
    lo = torch.stack((grid["xs"][0], grid["ys"][0]))
    hi = torch.stack((grid["xs"][-1], grid["ys"][-1]))
    valid &= torch.all((xy >= lo.view(1, 2)) & (xy <= hi.view(1, 2)), dim=-1)
    if initial_direction is not None:
        goal_direction = torch.nn.functional.normalize(xy - start_xy.view(1, 2), dim=-1)
        alignment = torch.sum(goal_direction * initial_direction.view(1, 2), dim=-1)
        aligned = valid & (alignment >= float(initial_min_alignment))
        # Preserve the initial-motion preference and its empty-set fallback.
        # The caller enforces the requested direction on the actual first step.
        if bool(torch.any(aligned).item()):
            valid = aligned
    candidates = free[valid]
    if candidates.numel() == 0:
        return None
    return candidates


def _route_tree(grid, occupied, start):
    """Reuse identical searches within a layout, including across route retries.

    Include the occupancy contents because the first segment and forced initial
    step use different masks. The cache dies with this scenario's planning grid.
    """
    cache = grid.setdefault("route_tree_cache", {})
    key = (tuple(occupied.shape), start, occupied.detach().cpu().numpy().tobytes())
    if key not in cache:
        cache[key] = grid_shortest_path_tree(occupied, start)
    return cache[key]


def polyline_length_xy(xy: torch.Tensor) -> float:
    if xy.shape[0] <= 1:
        return 0.0
    return float(torch.sum(torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1)).item())


def _bin_bounds(lo: float, hi: float, index: int, count: int) -> tuple[float, float]:
    count = max(1, int(count))
    index = min(max(0, int(index)), count - 1)
    width = (hi - lo) / count
    return lo + index * width, lo + (index + 1) * width


def sample_altitude(env, bucket: int) -> float:
    lo, hi = target.altitude_bounds(env)
    count = max(1, int(env.cfg.scenario.altitude_bin_count))
    lo, hi = _bin_bounds(lo, hi, bucket, count)
    return float(torch.empty((), device=env.device).uniform_(lo, hi).item())


def sample_target_cell(
    env,
    grid: dict,
    interceptor_xy: torch.Tensor,
    request: CoverageRequest,
) -> tuple[int, int] | None:
    free = grid_free_cells(grid)
    if free.numel() == 0:
        return None
    xy = grid_cells_xy(grid, free)
    delta = xy - interceptor_xy.view(1, 2)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    bins = tuple(env.cfg.scenario.start_distance_bins)
    if bins and request.distance_bin >= 0:
        lower, upper = bins[min(request.distance_bin, len(bins) - 1)]
    else:
        lower, upper = target.start_distance_min(env), float("inf")
    lower = max(target.start_distance_min(env), float(lower))
    valid = (distance >= lower) & (distance < float(upper))

    candidates = free[valid]
    if candidates.numel() == 0:
        return None
    pick = int(torch.randint(0, candidates.shape[0], (1,), device=env.device).item())
    return int(candidates[pick, 0]), int(candidates[pick, 1])


def sample_requested_speed(env, speed_bin: int) -> float:
    """Sample cruise speed from the phase quota's bin."""
    speed_range = env.cfg.scenario.target_speed_range
    speed_lo, speed_hi = sorted(float(value) for value in speed_range)
    edges = TARGET_SPEED_BIN_EDGES
    scale = (speed_hi - speed_lo) / (edges[-1] - edges[0])
    bin_lo = speed_lo + (edges[speed_bin] - edges[0]) * scale
    bin_hi = speed_lo + (edges[speed_bin + 1] - edges[0]) * scale
    return float(torch.empty((), device=env.device).uniform_(bin_lo, bin_hi).item()) if bin_hi > bin_lo else bin_lo


def sample_target_waypoints(
    env,
    grid: dict,
    start_cell: tuple[int, int],
    interceptor_start: torch.Tensor,
    z0: float,
    request: CoverageRequest,
    interceptor_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if request.motion_mode not in target.MOTION_MODE_ID:
        raise ValueError(f"Unsupported training motion mode: {request.motion_mode}")
    if request.motion_mode == "hover":
        return grid_static_evader_waypoints(env, grid, start_cell, z0)
    speed = sample_requested_speed(env, request.speed_bin)
    dt = (env._path_waypoint_steps[1:] - env._path_waypoint_steps[:-1]).to(torch.float32) * env._step_dt
    route_step = speed * dt
    start_xy = grid_cell_xy(grid, start_cell)
    away = start_xy - interceptor_start[:2]
    away = away / torch.linalg.vector_norm(away).clamp_min(1e-6)
    initial_motion = request.initial_motion
    if initial_motion == "random":
        start_distance = float(torch.linalg.vector_norm(start_xy - interceptor_start[:2]).item())
        choices = ("transverse", "away") if start_distance < target.min_start_distance_for_toward_motion(env) else (
            "toward", "transverse", "away"
        )
        initial_motion = choices[
            int(torch.randint(0, len(choices), (1,), device=env.device).item())
        ]
    if initial_motion == "toward":
        preferred = -away
    elif initial_motion == "transverse":
        preferred = torch.stack((-away[1], away[0]))
        if bool(torch.rand((), device=env.device) < 0.5):
            preferred = -preferred
    else:
        preferred = away

    min_initial_alignment = (
        math.sqrt(1.0 - 0.3**2) + 1e-3 if initial_motion == "transverse" else 0.3
    )

    base = sample_grid_evader_waypoints(
        env,
        grid,
        start_cell,
        z0,
        step_dist=route_step,
        preferred_initial_direction=preferred,
        preferred_initial_min_alignment=min_initial_alignment,
        # Cruise speed bounds the distance covered in this time, so the square
        # stays blocked for at least that long.
        avoid_mask=interceptor_mask,
        avoid_distance=speed * float(env.cfg.scenario.target_avoids_interceptor_start_s),
    )
    return base
