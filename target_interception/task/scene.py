"""Spawn arena walls and parked obstacle slots."""
from __future__ import annotations

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils


def spawn_arena_walls(num_envs: int, cfg) -> None:
    """Spawn the four arena walls (thin cuboids) around each env's arena."""
    span_x = (cfg.arena_max[0] - cfg.arena_min[0]) + 2 * cfg.wall_extra_margin
    span_y = (cfg.arena_max[1] - cfg.arena_min[1]) + 2 * cfg.wall_extra_margin
    height = cfg.arena_max[2] - cfg.arena_min[2]
    thickness = cfg.wall_thickness
    half_thickness = 0.5 * thickness

    center_z_local = 0.5 * (cfg.arena_max[2] + cfg.arena_min[2])

    x_pos_local = cfg.arena_max[0] + cfg.wall_extra_margin + half_thickness
    x_neg_local = cfg.arena_min[0] - cfg.wall_extra_margin - half_thickness
    y_pos_local = cfg.arena_max[1] + cfg.wall_extra_margin + half_thickness
    y_neg_local = cfg.arena_min[1] - cfg.wall_extra_margin - half_thickness

    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.04, 0.04, 0.04))

    wall_x_cfg = sim_utils.CuboidCfg(size=(thickness, span_y, height), visual_material=material, copy_from_source=False)
    wall_y_cfg = sim_utils.CuboidCfg(size=(span_x, thickness, height), visual_material=material, copy_from_source=False)

    for env_id in range(num_envs):
        base = f"/World/envs/env_{env_id}/Walls"
        placements = [
            (f"{base}/WallXPos", (x_pos_local, 0.0, center_z_local)),
            (f"{base}/WallXNeg", (x_neg_local, 0.0, center_z_local)),
            (f"{base}/WallYPos", (0.0, y_pos_local, center_z_local)),
            (f"{base}/WallYNeg", (0.0, y_neg_local, center_z_local)),
        ]
        for path, translation in placements:
            if prim_utils.is_prim_path_valid(path):
                continue
            wall_cfg = wall_x_cfg if "WallX" in path else wall_y_cfg
            wall_cfg.func(path, wall_cfg, translation=translation)


def _parked_xy(cfg, slot: int, offset: float) -> tuple[float, float]:
    """Out-of-arena parking spot for an obstacle slot (it is moved into place at reset)."""
    return (cfg.arena_min[0] - offset - 0.35 * slot, cfg.arena_min[1] - offset)


def spawn_static_obstacle_slots(num_envs: int, cfg, start_slot: int = 0, end_slot: int = 0) -> None:
    """Spawn interception static-pillar slots as parked placeholders (moved via XformPrimView)."""
    start_slot = max(0, int(start_slot))
    end_slot = int(end_slot)
    if end_slot <= start_slot:
        return

    center_z_local = cfg.arena_min[2] + 0.5 * cfg.pillar_height
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18))
    pillar_cfg = sim_utils.CylinderCfg(
        radius=cfg.pillar_radius,
        height=cfg.pillar_height,
        visual_material=material,
        copy_from_source=False,
    )
    for env_id in range(num_envs):
        base = f"/World/envs/env_{env_id}/Pillars"
        for slot in range(start_slot, end_slot):
            path = f"{base}/Pillar{slot}"
            if prim_utils.is_prim_path_valid(path):
                continue
            x_local, y_local = _parked_xy(cfg, slot, 6.0)
            pillar_cfg.func(path, pillar_cfg, translation=(x_local, y_local, center_z_local))


def spawn_dynamic_obstacle_slots(num_envs: int, cfg, start_slot: int = 0, end_slot: int = 0) -> None:
    """Spawn interception dynamic-obstacle slots as parked placeholders (moved via XformPrimView)."""
    start_slot = max(0, int(start_slot))
    end_slot = int(end_slot)
    if end_slot <= start_slot:
        return

    center_z_local = cfg.arena_min[2] + 0.5 * cfg.dynamic_obstacle_height
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.12, 0.08))
    obstacle_cfg = sim_utils.CylinderCfg(
        radius=cfg.pillar_radius,
        height=cfg.dynamic_obstacle_height,
        visual_material=material,
        copy_from_source=False,
    )
    for env_id in range(num_envs):
        base = f"/World/envs/env_{env_id}/DynamicObstacles"
        for slot in range(start_slot, end_slot):
            path = f"{base}/Dynamic{slot}"
            if prim_utils.is_prim_path_valid(path):
                continue
            x_local, y_local = _parked_xy(cfg, slot, 7.5)
            obstacle_cfg.func(path, obstacle_cfg, translation=(x_local, y_local, center_z_local))
