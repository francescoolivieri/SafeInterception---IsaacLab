"""Occupancy grid and shortest paths for scenario generation."""
from __future__ import annotations

import heapq
import math

import torch

from target_interception.task import target


def make_grid(env, *, target_bounds: bool = True, cell_size: float | None = None) -> dict | None:
    cell = max(0.05, float(env.cfg.scenario.grid_cell_size if cell_size is None else cell_size))

    margin = max(0.0, float(env.cfg.scenario.grid_wall_margin), target.wall_clearance(env))
    lo = env._arena_min_safe[:2] + margin
    hi = env._arena_max_safe[:2] - margin
    if target_bounds and env.cfg.scenario.target_wall_surface_clearance_xy is not None:
        target_lo, target_hi = target.arena_bounds(env)
        lo, hi = target_lo[:2], target_hi[:2]
    if bool(torch.any(hi <= lo).item()):
        return None

    xs = torch.arange(float(lo[0]), float(hi[0]) + 0.5 * cell, cell, device=env.device)
    ys = torch.arange(float(lo[1]), float(hi[1]) + 0.5 * cell, cell, device=env.device)
    xs = xs[xs <= float(hi[0]) + 1e-6]
    ys = ys[ys <= float(hi[1]) + 1e-6]
    if xs.numel() < 3 or ys.numel() < 3:
        return None

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    occupied = torch.zeros((ys.numel(), xs.numel()), dtype=torch.bool, device=env.device)
    # Planning includes target clearance; placement uses obstacle bodies and rails.
    # Reusing the planning mask would count the target margin twice.
    return {
        "xs": xs,
        "ys": ys,
        "xx": xx,
        "yy": yy,
        "occupied": occupied,
        "placement": occupied.clone(),
        "corridor": occupied.clone(),
        "cell_size": cell,
    }


def grid_blocked(grid: dict) -> torch.Tensor:
    """Cells unavailable for obstacle placement (obstacle bodies + swept rails)."""
    return grid["placement"] | grid["corridor"]


def grid_free_cells(grid: dict) -> torch.Tensor:
    return torch.nonzero(~grid["occupied"], as_tuple=False)


def obstacle_xy_bounds(env, grid: dict) -> tuple[torch.Tensor, torch.Tensor]:
    margin = max(
        float(env.cfg.scenario.grid_wall_margin),
        float(env.cfg.scenario.obstacle_wall_margin),
    )
    lo = torch.maximum(env._arena_min_safe[:2] + margin, torch.stack((grid["xs"][0], grid["ys"][0])))
    hi = torch.minimum(env._arena_max_safe[:2] - margin, torch.stack((grid["xs"][-1], grid["ys"][-1])))
    return lo, hi


def grid_obstacle_candidate_cells(env, grid: dict) -> torch.Tensor:
    lo, hi = obstacle_xy_bounds(env, grid)
    if bool(torch.any(hi <= lo).item()):
        return torch.empty((0, 2), device=env.device, dtype=torch.long)
    free = torch.nonzero(~grid_blocked(grid), as_tuple=False)
    if free.numel() == 0:
        return free
    xy = grid_cells_xy(grid, free)
    return free[torch.all((xy >= lo.view(1, 2)) & (xy <= hi.view(1, 2)), dim=-1)]


def grid_cell_xy(grid: dict, cell: tuple[int, int]) -> torch.Tensor:
    row, col = cell
    return torch.stack((grid["xs"][col], grid["ys"][row]))


def grid_cells_xy(grid: dict, cells: torch.Tensor) -> torch.Tensor:
    return torch.stack((grid["xs"][cells[:, 1]], grid["ys"][cells[:, 0]]), dim=-1)


def grid_cells_to_xy(env, grid: dict, cells: list[tuple[int, int]]) -> torch.Tensor:
    rows = torch.tensor([cell[0] for cell in cells], device=env.device, dtype=torch.long)
    cols = torch.tensor([cell[1] for cell in cells], device=env.device, dtype=torch.long)
    return torch.stack((grid["xs"][cols], grid["ys"][rows]), dim=-1)


def mark_grid_square(grid: dict, center: torch.Tensor, side: float) -> torch.Tensor:
    half = 0.5 * max(0.0, float(side))
    return (torch.abs(grid["xx"] - center[0]) <= half) & (torch.abs(grid["yy"] - center[1]) <= half)


def grid_disc_mask(grid: dict, center: torch.Tensor, radius: float) -> torch.Tensor:
    dx = grid["xx"] - center[0]
    dy = grid["yy"] - center[1]
    return dx * dx + dy * dy <= float(radius) ** 2


def grid_shortest_path_tree(occupied: torch.Tensor, start: tuple[int, int]) -> tuple[list[int], int]:
    """One Dijkstra search serves all candidate goals from the same start."""
    occ = occupied.detach().cpu().flatten().tolist()
    rows, cols = (int(value) for value in occupied.shape)
    size = rows * cols
    parent = [-1] * size
    if size == 0:
        return parent, cols

    source = start[0] * cols + start[1]
    if occ[source]:
        return parent, cols

    moves = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    cost = [float("inf")] * size
    cost[source] = 0.0
    closed = bytearray(size)
    heap = [(0.0, 0, source)]
    push_id = 1
    while heap:
        current_cost, _, cell = heapq.heappop(heap)
        if closed[cell]:
            continue
        closed[cell] = 1
        row, col = divmod(cell, cols)
        for dr, dc, move_cost in moves:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            nxt = nr * cols + nc
            if occ[nxt] or (dr != 0 and dc != 0 and (occ[row * cols + nc] or occ[nr * cols + col])):
                continue
            new_cost = current_cost + move_cost
            if new_cost >= cost[nxt]:
                continue
            cost[nxt] = new_cost
            parent[nxt] = cell
            heapq.heappush(heap, (new_cost, push_id, nxt))
            push_id += 1
    return parent, cols


def grid_tree_path(
    tree: tuple[list[int], int], start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """Recover one goal path from grid_shortest_path_tree."""
    parent, cols = tree
    source = start[0] * cols + start[1]
    cell = goal[0] * cols + goal[1]
    if cell != source and (cell >= len(parent) or parent[cell] < 0):
        return None
    path = []
    while True:
        path.append(divmod(cell, cols))
        if cell == source:
            return list(reversed(path))
        cell = parent[cell]
        if cell < 0:
            return None


def grid_tree_reachable_cells(
    tree: tuple[list[int], int], cells: torch.Tensor
) -> torch.Tensor:
    """Keep goals connected to the tree source.

    Candidates exclude the source, so a nonnegative parent means reachable."""
    if cells.numel() == 0:
        return cells
    parent, cols = tree
    reachable = torch.tensor(
        [parent[int(row) * cols + int(col)] >= 0 for row, col in cells.tolist()],
        dtype=torch.bool,
        device=cells.device,
    )
    return cells[reachable]
