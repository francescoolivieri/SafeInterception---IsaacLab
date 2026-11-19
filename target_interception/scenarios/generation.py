"""Sample obstacle layouts and target routes; validate their executed motion."""
from __future__ import annotations

import math

import torch

from target_interception.task import target

from . import routes
from .generation_stats import count as count_generation
from .generation_stats import time_stage
from .grid import (
    grid_blocked,
    grid_cell_xy,
    grid_disc_mask,
    grid_obstacle_candidate_cells,
    make_grid,
    mark_grid_square,
    obstacle_xy_bounds,
)


def static_grid_radius(env) -> float:
    return float(
        env.cfg.pillar_radius
        + target.target_radius(env)
        + env.cfg.scenario.obstacle_clearance
    )


def static_placement_radius(env) -> float:
    """Half of the configured pillar-to-pillar centre clearance."""
    return float(env.cfg.pillar_radius + 0.5 * env.cfg.scenario.obstacle_clearance)


# Inactive-slot parking + scene-z helpers (shared with the env's obstacle moves)

def inactive_obstacle_xy(env, slots: int) -> torch.Tensor:
    idx = torch.arange(max(0, slots), device=env.device, dtype=torch.float32)
    x = env._arena_min_safe[0] - 6.0 - 0.35 * idx
    y = torch.full_like(x, float(env._arena_min_safe[1]) - 6.0)
    return torch.stack((x, y), dim=-1)


def inactive_dynamic_pos(env, slots: int) -> torch.Tensor:
    xy = inactive_obstacle_xy(env, slots)
    z = torch.full((xy.shape[0], 1), dynamic_center_z(env), device=env.device)
    return torch.cat((xy, z), dim=-1)


def static_center_z(env) -> float:
    return float(env.cfg.arena_min[2] + 0.5 * env.cfg.pillar_height)


def dynamic_center_z(env) -> float:
    return float(env.cfg.arena_min[2] + 0.5 * env.cfg.dynamic_obstacle_height)


# Obstacle + pursuer sampling

def sample_pursuer_start(env, grid: dict) -> torch.Tensor | None:
    # Target wall access must not move the interceptor start distribution.
    if env.cfg.scenario.target_wall_surface_clearance_xy is not None:
        grid = make_grid(env, target_bounds=False)
        if grid is None:
            return None
    rows = int(grid["ys"].shape[0])
    cols = int(grid["xs"].shape[0])
    if rows < 1 or cols < 1:
        return None

    side = int(torch.randint(0, 4, (1,), device=env.device).item())
    if side == 0:
        cell = (int(torch.randint(0, rows, (1,), device=env.device).item()), 0)
    elif side == 1:
        cell = (int(torch.randint(0, rows, (1,), device=env.device).item()), cols - 1)
    elif side == 2:
        cell = (0, int(torch.randint(0, cols, (1,), device=env.device).item()))
    else:
        cell = (rows - 1, int(torch.randint(0, cols, (1,), device=env.device).item()))

    pos = torch.zeros(3, device=env.device)
    pos[:2] = grid_cell_xy(grid, cell)
    pos[2] = 0.5 * (env._arena_min_safe[2] + env._arena_max_safe[2])
    return pos


def sample_dynamic_obstacles(
    env,
    grid: dict,
    count: int,
    *,
    evader_waypoints: torch.Tensor | None = None,
    pursuer_xy: torch.Tensor | None = None,
    static_xy: torch.Tensor | None = None,
    static_active: torch.Tensor | None = None,
    length_bins: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Place moving obstacles, retrying each rail independently.

    Check target separation and interceptor spawn clearance when supplied."""
    slots = env._max_dynamic_obstacles
    waypoints = inactive_dynamic_pos(env, slots).view(1, slots, 3).repeat(env._path_waypoint_count, 1, 1)
    active = torch.zeros(slots, dtype=torch.bool, device=env.device)
    count = min(max(0, int(count)), slots)
    if count == 0:
        return waypoints, active

    patterns = tuple(env.cfg.scenario.dynamic_motion_patterns) or ("rail",)
    tries = max(1, int(env.cfg.scenario.dynamic_obstacle_retries))
    clearance = target_dynamic_clearance(env)
    pursuer_safe = float(env.cfg.pillar_radius + env.cfg.drone_collision_radius)
    obstacle_safe = 2.0 * float(env.cfg.pillar_radius) + float(
        env.cfg.scenario.obstacle_clearance
    )
    one_active = torch.ones(1, dtype=torch.bool, device=env.device)
    configured_bins = tuple(
        env.cfg.scenario.dynamic_rail_length_bins
    )
    for slot in range(count):
        placed = False
        remaining = tries
        while remaining > 0 and not placed:
            requested_bin = int(length_bins[slot]) if slot < len(length_bins) else int(
                torch.randint(0, len(configured_bins), (1,), device=env.device).item()
            )
            requested_bin = min(max(0, requested_bin), len(configured_bins) - 1)
            rail_range = configured_bins[requested_bin]
            rails = sample_dynamic_rail_candidates(
                env, grid, max(96, 4 * remaining), length_range=rail_range
            )
            if rails is None:
                count_generation(env, "dynamic.no_rail_candidates")
                return None
            starts, ends, masks = rails
            batch = min(remaining, int(starts.shape[0]))
            remaining -= batch
            for candidate in range(batch):
                count_generation(env, "dynamic.rail_attempts")
                start_xy, end_xy, mask = starts[candidate], ends[candidate], masks[candidate]
                pattern = patterns[int(torch.randint(0, len(patterns), (1,), device=env.device).item())]
                stop_go = sample_stop_go(env) if pattern == "stop_and_go" else None
                speed_range = env.cfg.scenario.dynamic_speed_range
                speed_lo, speed_hi = sorted(float(value) for value in speed_range)
                speed = float(torch.empty((), device=env.device).uniform_(speed_lo, speed_hi).item())
                position_phase = float(torch.rand((), device=env.device).item())
                travel_direction = int(torch.randint(0, 2, (1,), device=env.device).item())
                phase0 = position_phase if travel_direction == 0 else 2.0 - position_phase
                slot_waypoints = dynamic_rail_waypoints(
                    env,
                    start_xy,
                    end_xy,
                    pattern=pattern,
                    phase0=phase0,
                    stop_go=stop_go,
                    speed=speed,
                )
                rail_length = float(torch.linalg.vector_norm(end_xy - start_xy).item())
                rail_direction = (end_xy - start_xy) / max(rail_length, 1e-6)
                projected = torch.sum(
                    (slot_waypoints[:, :2] - start_xy.view(1, 2)) * rail_direction.view(1, 2),
                    dim=-1,
                )
                effective_span = float((projected.max() - projected.min()).item())
                if effective_span + 1e-6 < float(
                    env.cfg.scenario.dynamic_min_effective_span_fraction
                ) * rail_length:
                    count_generation(env, "dynamic.effective_span")
                    continue
                if pursuer_xy is not None:
                    if float(torch.linalg.vector_norm(slot_waypoints[0, :2] - pursuer_xy).item()) <= pursuer_safe:
                        count_generation(env, "dynamic.interceptor_clearance")
                        continue
                if static_xy is not None and static_active is not None and bool(static_active.any().item()):
                    active_static = static_xy[static_active]
                    rel = slot_waypoints[:, None, :2] - active_static[None, :, :]
                    separation = polyline_min_origin_distance(rel.movedim(1, 0))
                    if bool(torch.any(separation < obstacle_safe).item()):
                        count_generation(env, "dynamic.static_clearance")
                        continue
                if slot > 0:
                    rel = slot_waypoints[:, None, :2] - waypoints[:, :slot, :2]
                    separation = polyline_min_origin_distance(rel.movedim(1, 0))
                    if bool(torch.any(separation < obstacle_safe).item()):
                        count_generation(env, "dynamic.dynamic_clearance")
                        continue
                if evader_waypoints is not None:
                    separation = executed_target_dynamic_min_separation(
                        env, evader_waypoints, slot_waypoints.unsqueeze(1), one_active
                    )
                    if float(separation.item()) < clearance:
                        count_generation(env, "dynamic.target_collision")
                        continue
                waypoints[:, slot] = slot_waypoints
                active[slot] = True
                # Corridor, not occupied: later rails and pillars keep out of the
                # swept space, but target crossings are allowed when they remain
                # separated in time.
                grid["corridor"] |= mask
                placed = True
                count_generation(env, "dynamic.rail_accepted")
                break
        if not placed:
            count_generation(env, "dynamic.slot_exhausted")
            return None
    return waypoints, active


def sample_dynamic_rail_candidates(
    env,
    grid: dict,
    count: int = 96,
    *,
    length_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return valid iid rail candidates in their sampling order."""
    if length_range is None:
        length_range = tuple(
            env.cfg.scenario.dynamic_rail_length_range
        )
    length_lo = max(0.1, float(min(length_range)))
    length_hi = max(length_lo, float(max(length_range)))
    # Broad-phase rail radius; exact trajectory checks enforce the final clearance.
    radius = static_placement_radius(env)
    lo, hi = obstacle_xy_bounds(env, grid)

    free = grid_obstacle_candidate_cells(env, grid)
    if free.numel() == 0:
        return None
    blocked = grid_blocked(grid)

    n = max(1, int(count))
    device = env.device
    angle = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    centers = lo.view(1, 2) + torch.rand(n, 2, device=device) * (hi - lo).view(1, 2)
    length = torch.empty(n, 1, device=device).uniform_(length_lo, length_hi)
    starts = centers - 0.5 * length * direction
    ends = centers + 0.5 * length * direction
    in_bounds = torch.all(
        (starts >= lo) & (starts <= hi) & (ends >= lo) & (ends <= hi), dim=-1
    )
    # Reject out-of-bounds rails before rasterizing, preserving proposal order.
    starts, ends = starts[in_bounds], ends[in_bounds]
    if starts.shape[0] == 0:
        return None

    # Batched distance-to-segment tests on the placement grid.
    points = torch.stack((grid["xx"], grid["yy"]), dim=-1)  # [H, W, 2]
    ab = ends - starts  # [n, 2]
    denom = torch.sum(ab * ab, dim=-1).clamp_min(1e-6)  # [n]
    rel = points.unsqueeze(0) - starts.view(-1, 1, 1, 2)  # [n, H, W, 2]
    t = (torch.sum(rel * ab.view(-1, 1, 1, 2), dim=-1) / denom.view(-1, 1, 1)).clamp(0.0, 1.0)
    closest = starts.view(-1, 1, 1, 2) + t.unsqueeze(-1) * ab.view(-1, 1, 1, 2)
    masks = torch.sum((points.unsqueeze(0) - closest) ** 2, dim=-1) <= float(radius) ** 2

    valid = ~(masks & blocked.unsqueeze(0)).flatten(1).any(dim=1)
    picks = torch.nonzero(valid, as_tuple=False).flatten()
    if picks.numel() == 0:
        return None
    return starts[picks], ends[picks], masks[picks]


def sample_dynamic_rail(env, grid: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Sample one rail (start, end, swept-corridor mask) on the placement grid."""
    candidates = sample_dynamic_rail_candidates(env, grid)
    if candidates is None:
        return None
    starts, ends, masks = candidates
    return starts[0], ends[0], masks[0]


def sample_stop_go(env) -> tuple[float, float]:
    """Per-obstacle (move, pause) seconds, each uniform in dynamic_stop_go_range."""
    lo, hi = (float(v) for v in env.cfg.scenario.dynamic_stop_go_range)
    lo, hi = max(1e-3, min(lo, hi)), max(1e-3, max(lo, hi))
    move, pause = (float(v.item()) for v in torch.empty(2, device=env.device).uniform_(lo, hi))
    return move, pause


def _stop_go_rail_time(t: torch.Tensor, move: float, pause: float, ramp: float) -> torch.Tensor:
    """Map wall time to time-at-full-speed for a move/pause cycle with speed ramps.

    Within each move window the speed is trapezoidal: ramp up over ramp seconds,
    cruise, ramp down over ramp seconds, then rest for pause seconds. Full-speed
    equivalent progress per cycle is move - ramp.
    """
    ramp = max(1e-3, min(float(ramp), move / 3.0))
    period = move + pause
    tau = torch.remainder(t, period)
    rise = 0.5 * tau.clamp(max=ramp) ** 2 / ramp
    cruise = tau.clamp(min=ramp, max=move - ramp) - ramp
    fall = 0.5 * (ramp**2 - (move - tau.clamp(min=move - ramp, max=move)) ** 2) / ramp
    return torch.floor(t / period) * (move - ramp) + rise + cruise + fall


def dynamic_rail_waypoints(
    env,
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    *,
    pattern: str = "rail",
    phase0: float = 0.0,
    stop_go: tuple[float, float] | None = None,
    speed: float | None = None,
) -> torch.Tensor:
    """Bake a moving obstacle's trajectory into waypoints.

    All modes shuttle along a rail. stop_and_go adds pauses; wobble adds a lateral
    sine wave. phase0 in [0, 2) sets the initial position and direction."""
    scn = env.cfg.scenario
    path = torch.zeros(env._path_waypoint_count, 3, device=env.device)
    length = torch.linalg.vector_norm(end_xy - start_xy).clamp_min(1e-6)
    if speed is None:
        speed = max(float(value) for value in scn.dynamic_speed_range)
    speed = max(0.0, float(speed))
    t = env._path_waypoint_steps.to(torch.float32) * env._step_dt
    if speed <= 1e-6:
        phase = float(phase0) % 2.0
        alpha0 = phase if phase <= 1.0 else 2.0 - phase
        alpha = torch.full((env._path_waypoint_count,), alpha0, device=env.device)
    else:
        t_rail = t
        if pattern == "stop_and_go":
            if stop_go is None:
                midpoint = 0.5 * sum(float(v) for v in scn.dynamic_stop_go_range)
                stop_go = (midpoint, midpoint)
            move, pause = (max(1e-3, float(v)) for v in stop_go)
            t_rail = _stop_go_rail_time(t, move, pause, float(scn.dynamic_stop_go_ramp))
        phase = torch.remainder(t_rail * speed / length + float(phase0), 2.0)
        alpha = torch.where(phase <= 1.0, phase, 2.0 - phase)
        # Parabolic caps give smooth reversals, stopping delta/2 short of each end.
        ramp_s = max(0.0, float(scn.dynamic_stop_go_ramp))
        delta = min(0.25, ramp_s * speed / float(length))
        if delta > 1e-6:
            d_top = torch.abs(phase - 1.0)
            d_bot = torch.minimum(phase, 2.0 - phase)
            alpha = torch.where(d_top < delta, 1.0 - 0.5 * (d_top * d_top + delta * delta) / delta, alpha)
            alpha = torch.where(d_bot < delta, 0.5 * (d_bot * d_bot + delta * delta) / delta, alpha)
    xy = start_xy.view(1, 2) * (1.0 - alpha.view(-1, 1)) + end_xy.view(1, 2) * alpha.view(-1, 1)
    if pattern == "wobble":
        rail_dir = (end_xy - start_xy) / length
        perp = torch.stack((-rail_dir[1], rail_dir[0]))
        omega = 2.0 * math.pi / max(1e-3, float(scn.dynamic_wobble_period))
        offset = float(scn.dynamic_wobble_amplitude) * torch.sin(omega * t + math.pi * float(phase0))
        xy = xy + perp.view(1, 2) * offset.view(-1, 1)
    path[:, :2] = xy
    path[:, 2] = dynamic_center_z(env)
    return path


def sample_static_obstacles(env, grid: dict, count: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    slots = env._max_static_obstacles
    xy = inactive_obstacle_xy(env, slots)
    active = torch.zeros(slots, dtype=torch.bool, device=env.device)
    count = min(max(0, int(count)), slots)
    if count == 0:
        return xy, active

    planning_radius = static_grid_radius(env)
    placement_radius = static_placement_radius(env)
    safe_static = 2.0 * float(env.cfg.pillar_radius) + float(env.cfg.scenario.obstacle_clearance)
    for slot in range(count):
        free = grid_obstacle_candidate_cells(env, grid)
        if free.numel() == 0:
            return None

        placed = False
        lo, hi = obstacle_xy_bounds(env, grid)
        candidates = lo.view(1, 2) + torch.rand(128, 2, device=env.device) * (hi - lo).view(1, 2)
        for candidate in candidates:
            if slot > 0:
                dist = torch.linalg.vector_norm(xy[:slot] - candidate, dim=-1)
                if bool(torch.any(dist <= safe_static).item()):
                    continue
            placement_mask = grid_disc_mask(grid, candidate, placement_radius)
            if bool(torch.any(grid_blocked(grid) & placement_mask).item()):
                continue
            xy[slot] = candidate
            active[slot] = True
            grid["placement"] |= placement_mask
            # Cell centres use the exact clearance radius. A step between two free
            # centres may dip slightly inside it; the spline fit and final audit
            # enforce the continuous clearance on the executed reference.
            grid["occupied"] |= grid_disc_mask(grid, candidate, planning_radius)
            placed = True
            break
        if not placed:
            return None
    return xy, active


def point_free(env, pos, static_xy, static_active, dynamic_pos, dynamic_active) -> bool:
    if bool(static_active.any().item()):
        d_static = torch.linalg.vector_norm(static_xy - pos[:2], dim=-1)
        safe = float(env.cfg.pillar_radius + env.cfg.drone_collision_radius + env.cfg.scenario.obstacle_clearance)
        if bool(torch.any((d_static <= safe) & static_active).item()):
            return False
    if bool(dynamic_active.any().item()):
        dynamic_now = dynamic_pos[0] if dynamic_pos.ndim == 3 else dynamic_pos
        d_dyn = torch.linalg.vector_norm(dynamic_now[:, :2] - pos[:2], dim=-1)
        safe = float(env.cfg.pillar_radius + env.cfg.drone_collision_radius)
        if bool(((d_dyn <= safe) & dynamic_active).any().item()):
            return False
    return True


# Space-time evader <-> moving-obstacle separation

def target_dynamic_clearance(env) -> float:
    """Horizontal target-to-cylinder clearance, including the planning margin."""

    scn = env.cfg.scenario
    return float(
        env.cfg.pillar_radius
        + target.target_radius(env)
        + scn.obstacle_clearance
    )


def polyline_min_origin_distance(rel: torch.Tensor) -> torch.Tensor:
    """Min distance from the origin to the piecewise-linear XY path rel[..., W, 2]."""
    if rel.shape[-2] == 1:
        return torch.linalg.vector_norm(rel[..., 0, :], dim=-1)
    a = rel[..., :-1, :]
    d = rel[..., 1:, :] - a
    t = (-(a * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * d
    return torch.linalg.vector_norm(closest, dim=-1).min(dim=-1).values


def target_dynamic_min_separation(
    evader_waypoints: torch.Tensor, dynamic_waypoints: torch.Tensor, dynamic_active: torch.Tensor
) -> torch.Tensor:
    """Minimum same-time XY separation from active moving obstacles.

    Exact for linear paths on shared steps. Use executed_target_schedules for cubic
    targets. Accepts a single row or batch; returns infinity if no slot is active."""
    squeeze = evader_waypoints.ndim == 2
    ev = evader_waypoints.unsqueeze(0) if squeeze else evader_waypoints
    dyn = dynamic_waypoints.unsqueeze(0) if squeeze else dynamic_waypoints
    act = dynamic_active.unsqueeze(0) if squeeze else dynamic_active
    if act.shape[-1] == 0:
        out = ev.new_full((ev.shape[0],), float("inf"))
        return out[0] if squeeze else out
    rel = dyn[..., :2] - ev[:, :, None, :2]  # [k, W, slots, 2]
    dist = polyline_min_origin_distance(rel.movedim(2, 1))  # [k, slots]
    dist = torch.where(act, dist, torch.full_like(dist, float("inf")))
    out = dist.min(dim=-1).values
    return out[0] if squeeze else out




def executed_target_schedules(env, target_waypoints: torch.Tensor, dynamic_waypoints: torch.Tensor | None = None):
    """Sample target and obstacle motion on the policy clock for clearance checks.

    The target uses cubic interpolation; moving obstacles use linear interpolation."""

    executed = target.sample_policy_steps(env, target_waypoints, cubic=True, knot_dim=-2)
    if dynamic_waypoints is not None:
        dynamic_waypoints = target.sample_policy_steps(env, dynamic_waypoints, cubic=False, knot_dim=-3)
    return executed, dynamic_waypoints


def executed_target_dynamic_min_separation(
    env, target_waypoints: torch.Tensor, dynamic_waypoints: torch.Tensor, dynamic_active: torch.Tensor
) -> torch.Tensor:
    """target_dynamic_min_separation of the executed target motion."""
    executed, dynamic = executed_target_schedules(env, target_waypoints, dynamic_waypoints)
    return target_dynamic_min_separation(executed, dynamic, dynamic_active)


def point_dynamic_min_separation(
    point_xy: torch.Tensor, dynamic_waypoints: torch.Tensor, dynamic_active: torch.Tensor
) -> torch.Tensor:
    """Min-over-time XY distance from a fixed point to every active moving obstacle."""
    squeeze = point_xy.ndim == 1
    p = point_xy.unsqueeze(0) if squeeze else point_xy
    dyn = dynamic_waypoints.unsqueeze(0) if squeeze else dynamic_waypoints
    act = dynamic_active.unsqueeze(0) if squeeze else dynamic_active
    if act.shape[-1] == 0:
        out = p.new_full((p.shape[0],), float("inf"))
        return out[0] if squeeze else out
    rel = dyn[..., :2] - p[:, None, None, :]  # [k, W, slots, 2]
    dist = polyline_min_origin_distance(rel.movedim(2, 1))  # [k, slots]
    dist = torch.where(act, dist, torch.full_like(dist, float("inf")))
    out = dist.min(dim=-1).values
    return out[0] if squeeze else out


def obstacle_center_clearance(env) -> float:
    """Required centre separation for any pair of cylindrical obstacles."""
    return 2.0 * float(env.cfg.pillar_radius) + float(env.cfg.scenario.obstacle_clearance)


def dynamic_static_min_separation(
    dynamic_waypoints: torch.Tensor,
    dynamic_active: torch.Tensor,
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
) -> torch.Tensor:
    """Exact continuous-time minimum between moving and static obstacle centres."""
    dynamic = dynamic_waypoints[:, dynamic_active, :2]
    static = static_xy[static_active]
    if dynamic.shape[1] == 0 or static.shape[0] == 0:
        return dynamic_waypoints.new_tensor(float("inf"))
    relative = dynamic[:, :, None, :] - static[None, None, :, :]
    distance = polyline_min_origin_distance(relative.permute(1, 2, 0, 3))
    return distance.min()


def dynamic_dynamic_min_separation(
    dynamic_waypoints: torch.Tensor,
    dynamic_active: torch.Tensor,
) -> torch.Tensor:
    """Exact same-time minimum between every pair of moving obstacles."""
    dynamic = dynamic_waypoints[:, dynamic_active, :2]
    count = int(dynamic.shape[1])
    if count < 2:
        return dynamic_waypoints.new_tensor(float("inf"))
    pair_distances = []
    for first in range(count - 1):
        relative = dynamic[:, first + 1 :, :] - dynamic[:, first : first + 1, :]
        pair_distances.append(polyline_min_origin_distance(relative.movedim(1, 0)))
    return torch.cat(pair_distances).min()


def static_static_min_separation(
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
) -> torch.Tensor:
    static = static_xy[static_active]
    if static.shape[0] < 2:
        return static_xy.new_tensor(float("inf"))
    return torch.pdist(static).min()


def final_obstacle_layout_valid(
    env,
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
    dynamic_waypoints: torch.Tensor,
    dynamic_active: torch.Tensor,
) -> bool:
    """Continuous geometry gate applied to every completed obstacle layout."""
    if not bool(torch.isfinite(static_xy[static_active]).all().item()):
        return False
    if not bool(torch.isfinite(dynamic_waypoints[:, dynamic_active]).all().item()):
        return False
    safe = obstacle_center_clearance(env)
    separations = (
        static_static_min_separation(static_xy, static_active),
        dynamic_static_min_separation(dynamic_waypoints, dynamic_active, static_xy, static_active),
        dynamic_dynamic_min_separation(dynamic_waypoints, dynamic_active),
    )
    if any(float(value.item()) < safe - 1e-6 for value in separations):
        return False
    lo = env._arena_min_safe[:2] + float(env.cfg.pillar_radius)
    hi = env._arena_max_safe[:2] - float(env.cfg.pillar_radius)
    if bool(static_active.any().item()):
        static = static_xy[static_active]
        if not bool(torch.all((static >= lo) & (static <= hi)).item()):
            return False
    if bool(dynamic_active.any().item()):
        dynamic = dynamic_waypoints[:, dynamic_active]
        if not bool(torch.all((dynamic[..., :2] >= lo) & (dynamic[..., :2] <= hi)).item()):
            return False
        expected_z = dynamic_center_z(env)
        if not bool(torch.all(torch.abs(dynamic[..., 2] - expected_z) <= 1e-5).item()):
            return False
    return True


def target_static_min_separation(
    target_waypoints: torch.Tensor,
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
) -> torch.Tensor:
    """Exact continuous XY segment distance to every active vertical pillar."""
    squeeze = target_waypoints.ndim == 2
    target_wp = target_waypoints.unsqueeze(0) if squeeze else target_waypoints
    static = static_xy.unsqueeze(0) if squeeze else static_xy
    active = static_active.unsqueeze(0) if squeeze else static_active
    if active.shape[-1] == 0:
        out = target_wp.new_full((target_wp.shape[0],), float("inf"))
        return out[0] if squeeze else out
    rel = target_wp[:, :, None, :2] - static[:, None, :, :]
    distance = polyline_min_origin_distance(rel.movedim(2, 1))
    distance = torch.where(active, distance, torch.full_like(distance, float("inf")))
    out = distance.min(dim=-1).values
    return out[0] if squeeze else out


def target_trajectory_min_clearance(
    env,
    target_waypoints: torch.Tensor,
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
    dynamic_waypoints: torch.Tensor,
    dynamic_active: torch.Tensor,
) -> torch.Tensor:
    """Signed clearance of a finalized target reference from all clutter."""
    target_waypoints, dynamic_waypoints = executed_target_schedules(env, target_waypoints, dynamic_waypoints)
    static = target_static_min_separation(target_waypoints, static_xy, static_active) - static_grid_radius(env)
    dynamic = target_dynamic_min_separation(target_waypoints, dynamic_waypoints, dynamic_active)
    dynamic -= target_dynamic_clearance(env)
    return torch.minimum(static, dynamic)


def target_trajectory_in_arena(env, waypoints: torch.Tensor) -> bool:
    """Final wall/floor/ceiling check including body and tracking margins."""
    executed, _ = executed_target_schedules(env, waypoints)
    return _target_positions_in_arena(env, executed)


def _target_positions_in_arena(env, positions: torch.Tensor) -> bool:

    lo, hi = target.arena_bounds(env)
    # Grid cells may sit exactly on a bound; the cubic reconstruction of a held
    # point can then land a float32 ulp outside it.
    tol = 1e-5
    return bool(torch.all((positions >= lo.view(1, 3) - tol) & (positions <= hi.view(1, 3) + tol)).item())


def final_target_trajectory_valid(
    env,
    waypoints: torch.Tensor,
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
    dynamic_waypoints: torch.Tensor,
    dynamic_active: torch.Tensor,
) -> bool:
    """Validate the final runtime waypoints, after every smoothing/motion edit."""

    valid = True
    with time_stage(env, "reference_validation"):
        executed, dynamic_executed = executed_target_schedules(env, waypoints, dynamic_waypoints)
        finite = bool(torch.isfinite(waypoints).all().item())
        if not finite:
            count_generation(env, "final.nonfinite")
            valid = False
        elif not _target_positions_in_arena(env, executed):
            count_generation(env, "final.wall_violation")
            valid = False

        static_clearance = target_static_min_separation(
            executed, static_xy, static_active
        ) - static_grid_radius(env)
        if float(static_clearance.item()) < -1e-6:
            count_generation(env, "final.static_collision")
            valid = False
        dynamic_clearance = target_dynamic_min_separation(
            executed, dynamic_executed, dynamic_active
        ) - target_dynamic_clearance(env)
        if float(dynamic_clearance.item()) < -1e-6:
            count_generation(env, "final.dynamic_collision")
            valid = False
        if not target.reference_kinematics_ok(env, waypoints):
            valid = False
    return valid


# Scenario assembly + resets

def sample_scenario(env, phase: int, *, coverage) -> dict | None:
    """Sample a scenario for one coverage request.

    Place pillars, fit the target route, then sample rails against that route."""

    count_generation(env, "scenario.attempts")

    n_static, n_dynamic = coverage.static_count, coverage.dynamic_count
    with time_stage(env, "layout"):
        grid = make_grid(env)
    if grid is None:
        count_generation(env, "scenario.layout_placement")
        return None

    with time_stage(env, "layout"):
        pursuer_start = sample_pursuer_start(env, grid)
    if pursuer_start is None:
        count_generation(env, "scenario.layout_placement")
        return None
    # Obstacles never use the interceptor spawn square; target routes avoid it
    # only early on (target_avoids_interceptor_start_s).
    pursuer_mask = mark_grid_square(
        grid, pursuer_start[:2], float(env.cfg.scenario.pursuer_occupied_side)
    )
    grid["placement"] |= pursuer_mask

    with time_stage(env, "layout"):
        static = sample_static_obstacles(env, grid, n_static)
    if static is None:
        count_generation(env, "scenario.layout_placement")
        return None
    static_xy, static_active = static
    # Exact geometry for spline fitting; A* still uses the conservative grid.
    lo, hi = target.arena_bounds(env)
    grid["spline_geometry"] = dict(
        lo=lo[:2].cpu().double().numpy(), hi=hi[:2].cpu().double().numpy(),
        centers=static_xy[static_active.bool()].cpu().double().numpy(),
        radius=static_grid_radius(env),
    )
    # Validate the cubic with the same clearance as the pool audit.
    grid["collision_geometry"] = grid["spline_geometry"]

    static_evader = coverage.motion_mode == "hover"
    target_tries = 1 if static_evader else max(1, int(env.cfg.scenario.target_path_retries))
    target_ready = False
    for _ in range(target_tries):
        evader_cell = routes.sample_target_cell(env, grid, pursuer_start[:2], coverage)
        if evader_cell is None:
            break
        evader_z = routes.sample_altitude(env, coverage.target_altitude_bin)
        # Interceptor and target starts are sampled independently in 3D.
        pursuer_start[2] = routes.sample_altitude(env, coverage.interceptor_altitude_bin)

        if static_evader:
            evader_waypoints = routes.grid_static_evader_waypoints(env, grid, evader_cell, evader_z)
        else:
            with time_stage(env, "route"):
                evader_waypoints = routes.sample_target_waypoints(
                    env, grid, evader_cell, pursuer_start, evader_z, coverage, pursuer_mask
                )
            if evader_waypoints is None:
                count_generation(env, "scenario.route_rejected")
                continue
        path_type = target.MOTION_MODE_ID[coverage.motion_mode]

        first_vel = target.first_velocity(env, evader_waypoints)
        initial_motion = target.initial_motion_bucket(pursuer_start, evader_waypoints, first_vel)
        start_distance_xy = float(
            torch.linalg.vector_norm(evader_waypoints[0, :2] - pursuer_start[:2]).item()
        )
        if (
            coverage.initial_motion == "random"
            and start_distance_xy < target.min_start_distance_for_toward_motion(env)
            and initial_motion == "toward"
        ):
            count_generation(env, "scenario.initial_motion")
            continue
        if coverage.initial_motion != "random" and initial_motion != coverage.initial_motion:
            count_generation(env, "scenario.initial_motion")
            continue
        target_ready = True
        break
    if not target_ready:
        count_generation(env, "scenario.target_retries_exhausted")
        return None

    # Retry rails against the finished route; restore space reserved by failed draws.
    corridor_base = grid["corridor"]
    dynamic_waypoints = dynamic_active = None
    for _ in range(3):
        grid["corridor"] = corridor_base.clone()
        with time_stage(env, "moving_obstacles"):
            dynamic = sample_dynamic_obstacles(
                env,
                grid,
                n_dynamic,
                evader_waypoints=evader_waypoints,
                pursuer_xy=pursuer_start[:2],
                static_xy=static_xy,
                static_active=static_active,
                length_bins=coverage.dynamic_length_bins,
            )
        if dynamic is not None:
            dynamic_waypoints, dynamic_active = dynamic
            break
    if dynamic_waypoints is None:
        count_generation(env, "scenario.dynamic_placement")
        return None

    # No end-of-scenario audit: each stage above enforces its own continuous
    # check, and pool.validate_scenario_pools re-audits every saved row.
    yaw = torch.atan2(first_vel[1], first_vel[0])
    scenario = {
        "phase": int(phase),
        "path_type": int(path_type),
        "target_waypoints": evader_waypoints,
        "target_yaw": float(yaw.item()),
        "interceptor_start": pursuer_start,
        "static_xy": static_xy,
        "static_active": static_active,
        "dynamic_waypoints": dynamic_waypoints,
        "dynamic_active": dynamic_active,
    }
    count_generation(env, "scenario.accepted")
    return scenario
