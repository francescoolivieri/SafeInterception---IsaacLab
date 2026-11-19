"""Training-only augmentation of pooled scenarios.

Mirror geometry, freeze the target, replace a pillar or respawn the interceptor.
Placement trials keep the pooled values when clearance checks fail. Rows are
copies of the pool, so in-place edits are safe."""
from __future__ import annotations
from target_interception.scenarios import generation

import torch



def _probability(env, name: str) -> float:
    return max(0.0, min(1.0, float(getattr(env.cfg.scenario, name, 0.0))))


def augment_pool_rows(env, rows: dict, *, phase: int | None = None) -> dict:
    """Apply all enabled augmentations (in a fixed order) to a drawn row batch.

    Order matters mildly: mirror first (pure symmetry), then freeze
    before the placement trials so validity is checked against the target's
    final motion, interceptor respawn last so it sees the final pillars.
    """
    k = int(rows["target_waypoints"].shape[0])
    if k == 0:
        return rows
    _mirror(env, rows, k)
    _freeze_evader(env, rows, k)
    _replace_one_static(env, rows, k)
    from target_interception.scenarios import penalty_curriculum

    # The fallback must already share the mirrored scenario's coordinate frame.
    constrained = phase is not None and penalty_curriculum.enabled(env.cfg)
    original_spawn = rows["interceptor_start"].clone() if constrained else None
    _respawn_pursuer(env, rows, k)
    if constrained:
        penalty_curriculum.constrain_reset(env, rows, phase, original_spawn)
    return rows


# Individual transformations

def _mirror(env, rows: dict, k: int) -> None:
    p = _probability(env, "augment_mirror_probability")
    if p <= 0.0:
        return
    device = env.device
    flip = torch.rand(k, 2, device=device) < p  # [:, 0] mirrors x, [:, 1] mirrors y
    if not bool(flip.any().item()):
        return

    lo = torch.tensor(env.cfg.arena_min[:2], device=device, dtype=torch.float32)
    hi = torch.tensor(env.cfg.arena_max[:2], device=device, dtype=torch.float32)
    center = 0.5 * (lo + hi)

    def reflected(xy: torch.Tensor) -> torch.Tensor:
        f = flip.view(k, *([1] * (xy.ndim - 2)), 2)
        return torch.where(f, 2.0 * center - xy, xy)

    rows["target_waypoints"][..., :2] = reflected(rows["target_waypoints"][..., :2])
    rows["interceptor_start"][:, :2] = reflected(rows["interceptor_start"][:, :2])
    rows["static_xy"] = reflected(rows["static_xy"])
    dyn_xy = rows["dynamic_waypoints"][..., :2]
    rows["dynamic_waypoints"][..., :2] = reflected(dyn_xy)


def _target_moves(rows: dict) -> torch.Tensor:
    """Identify motion from the geometry."""
    displacement = rows["target_waypoints"][:, 1:] - rows["target_waypoints"][:, :1]
    return torch.linalg.vector_norm(displacement, dim=-1).amax(dim=1) > 0.05


def _freeze_evader(env, rows: dict, k: int) -> None:
    p = _probability(env, "augment_evader_freeze_probability")
    if p <= 0.0:
        return
    device = env.device
    selected = (torch.rand(k, device=device) < p) & _target_moves(rows)
    if not bool(selected.any().item()):
        return

    start = rows["target_waypoints"][:, 0]  # [k, 3]
    # A frozen evader keeps its start point for the whole episode, so no moving
    # obstacle may ever sweep within clearance of it.
    separation = generation.point_dynamic_min_separation(
        start[:, :2], rows["dynamic_waypoints"], rows["dynamic_active"]
    )
    selected &= separation >= generation.target_dynamic_clearance(env)

    frozen = start.unsqueeze(1).expand_as(rows["target_waypoints"])
    rows["target_waypoints"] = torch.where(selected.view(k, 1, 1), frozen, rows["target_waypoints"])


def _replace_one_static(env, rows: dict, k: int) -> None:
    p = _probability(env, "augment_static_replace_probability")
    if p <= 0.0:
        return
    device = env.device
    active = rows["static_active"]
    selected = (torch.rand(k, device=device) < p) & active.any(dim=1)
    if not bool(selected.any().item()):
        return

    slots = active.shape[1]
    scores = torch.where(active, torch.rand(k, slots, device=device), torch.full((k, slots), -1.0, device=device))
    slot = torch.argmax(scores, dim=1)  # random active slot per env

    scn = env.cfg.scenario
    margin = max(float(scn.grid_wall_margin), float(getattr(scn, "obstacle_wall_margin", scn.grid_wall_margin)))
    lo = env._arena_min_safe[:2] + margin
    hi = env._arena_max_safe[:2] - margin
    if bool(torch.any(hi <= lo).item()):
        return
    candidate = lo + (hi - lo) * torch.rand(k, 2, device=device)

    arange = torch.arange(k, device=device)
    ok = selected.clone()

    # Other active pillars (same spacing rule as the generator).
    d_static = torch.linalg.vector_norm(rows["static_xy"] - candidate[:, None, :], dim=-1)
    others = active.clone()
    others[arange, slot] = False
    d_static = torch.where(others, d_static, torch.full_like(d_static, float("inf")))
    ok &= d_static.min(dim=1).values > 2.0 * float(env.cfg.pillar_radius) + float(scn.obstacle_clearance)

    # Pursuer spawn clearance (same rule as point_free).
    d_pursuer = torch.linalg.vector_norm(rows["interceptor_start"][:, :2] - candidate, dim=-1)
    ok &= d_pursuer > float(env.cfg.pillar_radius + env.cfg.drone_collision_radius + scn.obstacle_clearance)

    # The whole executed evader path (same clearance as the planning grid).
    executed, _ = generation.executed_target_schedules(env, rows["target_waypoints"])
    rel_evader = executed[:, :, :2] - candidate[:, None, :]
    ok &= generation.polyline_min_origin_distance(rel_evader) > generation.static_grid_radius(env)

    # Every moving-obstacle trajectory (wobble is baked into the waypoints).
    separation = generation.point_dynamic_min_separation(candidate, rows["dynamic_waypoints"], rows["dynamic_active"])
    ok &= separation > float(2.0 * env.cfg.pillar_radius + scn.obstacle_clearance)

    rows["static_xy"][arange[ok], slot[ok]] = candidate[ok]


def _respawn_pursuer(env, rows: dict, k: int) -> None:
    p = _probability(env, "augment_pursuer_respawn_probability")
    if p <= 0.0:
        return
    device = env.device
    selected = torch.rand(k, device=device) < p
    if not bool(selected.any().item()):
        return

    scn = env.cfg.scenario
    candidates = max(1, int(getattr(scn, "augment_pursuer_respawn_candidates", 4)))
    margin = max(float(scn.interceptor_wall_clearance), float(scn.grid_wall_margin))
    lo = env._arena_min_safe[:2] + margin
    hi = env._arena_max_safe[:2] - margin
    if bool(torch.any(hi <= lo).item()):
        return
    xy = lo + (hi - lo) * torch.rand(k, candidates, 2, device=device)  # [k, K, 2]

    # Static pillars at any time / moving obstacles at t=0 (same rules as point_free).
    valid = torch.ones((k, candidates), dtype=torch.bool, device=device)
    if rows["static_active"].shape[1] > 0:
        d_static = torch.linalg.vector_norm(xy[:, :, None, :] - rows["static_xy"][:, None, :, :], dim=-1)
        d_static = torch.where(
            rows["static_active"][:, None, :], d_static, torch.full_like(d_static, float("inf"))
        )
        safe_static = float(env.cfg.pillar_radius + env.cfg.drone_collision_radius + scn.obstacle_clearance)
        valid &= d_static.min(dim=-1).values > safe_static

    dyn_now = rows["dynamic_waypoints"][:, 0, :, :2]  # [k, slots, 2]
    if rows["dynamic_active"].shape[1] > 0:
        d_dyn = torch.linalg.vector_norm(xy[:, :, None, :] - dyn_now[:, None, :, :], dim=-1)
        d_dyn = torch.where(rows["dynamic_active"][:, None, :], d_dyn, torch.full_like(d_dyn, float("inf")))
        valid &= d_dyn.min(dim=-1).values > float(env.cfg.pillar_radius + env.cfg.drone_collision_radius)

    # Minimum starting distance from the evader.
    evader_start = rows["target_waypoints"][:, 0, :2]
    d_evader = torch.linalg.vector_norm(xy - evader_start[:, None, :], dim=-1)
    from . import target

    valid &= d_evader >= target.start_distance_min(env)

    # First valid candidate per env; keep the pooled (border) spawn when none pass.
    pick = valid.to(torch.int64).argmax(dim=1)
    accept = selected & valid.any(dim=1)
    chosen = xy[torch.arange(k, device=device), pick]
    rows["interceptor_start"][:, :2] = torch.where(accept.unsqueeze(-1), chosen, rows["interceptor_start"][:, :2])
