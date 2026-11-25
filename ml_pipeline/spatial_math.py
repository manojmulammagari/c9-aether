from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_ORDER: Tuple[str, ...] = (
    "atk_alive",
    "def_alive",
    "atk_control_ratio",
    "def_control_ratio",
    "op_alive_def",
    "op_exposed_area_ratio",
    "op_targets_in_sight",
    "atk_min_site_dist",
    "def_rotation_time_sec",
    "spike_planted",
    "time_remaining_sec",
    "econ_diff_norm",
)

# Clean geometric tactical map walls in [0, 100] x [0, 100] space (100% Riot IP compliant)
MAP_WALLS: Tuple[Tuple[float, float, float, float], ...] = (
    (10.0, 38.0, 34.0, 38.0),  # A-Site front partition
    (34.0, 14.0, 34.0, 30.0),  # A-Heaven / Tree divider
    (46.0, 26.0, 46.0, 62.0),  # Mid Left pillar wall
    (56.0, 34.0, 56.0, 72.0),  # Mid Right catwalk divider
    (68.0, 20.0, 68.0, 44.0),  # B-Site market wall
    (74.0, 52.0, 92.0, 52.0),  # B-Main chokepoint wall
    (22.0, 64.0, 40.0, 64.0),  # A-Main lobby barrier
)

SITE_A_ANCHOR: Tuple[float, float] = (24.0, 22.0)
GRID_STEPS: int = 20
WALL_PENALTY_UNITS: float = 35.0
OPERATOR_COST_CREDITS: int = 4700
PLAYER_RUN_SPEED_UNITS_PER_SEC: float = 5.4


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    team: str  # "ATK" or "DEF"
    x: float
    y: float
    alive: bool = True
    has_operator: bool = False
    loadout_value: float = 3900.0


@dataclass(frozen=True)
class SmokeOrb:
    x: float
    y: float
    radius: float = 7.5


def _is_valid_coord(val: Optional[float]) -> bool:
    return val is not None and isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val)


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """Exact 2D line-segment intersection test."""
    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True
    return False


def segment_intersects_circle(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, radius: float,
) -> bool:
    """Returns True if line segment AB passes through or inside smoke circle (cx, cy, radius)."""
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return math.hypot(ax - cx, ay - cy) <= radius

    t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(closest_x - cx, closest_y - cy) <= radius


def has_line_of_sight(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    walls: Sequence[Tuple[float, float, float, float]] = MAP_WALLS,
    smokes: Sequence[SmokeOrb] = (),
) -> bool:
    """Returns True only if ray AB is unobstructed by both map walls and active smokes."""
    for wx1, wy1, wx2, wy2 in walls:
        if segments_intersect(ax, ay, bx, by, wx1, wy1, wx2, wy2):
            return False
    for smoke in smokes:
        if segment_intersects_circle(ax, ay, bx, by, smoke.x, smoke.y, smoke.radius):
            return False
    return True


def count_wall_crossings(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    walls: Sequence[Tuple[float, float, float, float]] = MAP_WALLS,
) -> int:
    return sum(
        1 for wx1, wy1, wx2, wy2 in walls
        if segments_intersect(ax, ay, bx, by, wx1, wy1, wx2, wy2)
    )


def extract_round_features(
    players: Sequence[PlayerState],
    smokes: Sequence[SmokeOrb] = (),
    spike_planted: bool = False,
    time_remaining_sec: float = 45.0,
    walls: Sequence[Tuple[float, float, float, float]] = MAP_WALLS,
) -> Dict[str, float]:
    """
    Computes the exact 12-feature contract dictionary with wall-aware Voronoi control,
    raycast Operator sightline exposure, and NaN-safe telemetry sanitization.
    """
    valid_atk: List[PlayerState] = []
    valid_def: List[PlayerState] = []
    atk_econ = 0.0
    def_econ = 0.0

    for p in players:
        if not p.alive:
            continue
        if not (_is_valid_coord(p.x) and _is_valid_coord(p.y)):
            continue
        econ = p.loadout_value if _is_valid_coord(p.loadout_value) else 0.0
        if p.team == "ATK":
            valid_atk.append(p)
            atk_econ += econ
        elif p.team == "DEF":
            valid_def.append(p)
            def_econ += econ

    def_operators = [p for p in valid_def if p.has_operator]
    op_alive_def = 1.0 if len(def_operators) > 0 else 0.0

    atk_cells = 0
    def_cells = 0
    op_exposed_cells = 0
    total_cells = GRID_STEPS * GRID_STEPS
    cell_size = 100.0 / GRID_STEPS

    for gx_idx in range(GRID_STEPS):
        gx = (gx_idx + 0.5) * cell_size
        for gy_idx in range(GRID_STEPS):
            gy = (gy_idx + 0.5) * cell_size

            best_atk = float("inf")
            for p in valid_atk:
                d = math.hypot(gx - p.x, gy - p.y) + WALL_PENALTY_UNITS * count_wall_crossings(p.x, p.y, gx, gy, walls)
                if d < best_atk:
                    best_atk = d

            best_def = float("inf")
            for p in valid_def:
                d = math.hypot(gx - p.x, gy - p.y) + WALL_PENALTY_UNITS * count_wall_crossings(p.x, p.y, gx, gy, walls)
                if d < best_def:
                    best_def = d

            if best_atk < best_def:
                atk_cells += 1
            elif best_def < best_atk:
                def_cells += 1

            if def_operators:
                if any(has_line_of_sight(op.x, op.y, gx, gy, walls, smokes) for op in def_operators):
                    op_exposed_cells += 1

    op_targets = 0
    if def_operators:
        for atk in valid_atk:
            if any(has_line_of_sight(op.x, op.y, atk.x, atk.y, walls, smokes) for op in def_operators):
                op_targets += 1

    if valid_atk:
        atk_min_site_dist = min(
            math.hypot(p.x - SITE_A_ANCHOR[0], p.y - SITE_A_ANCHOR[1]) for p in valid_atk
        )
    else:
        atk_min_site_dist = 100.0

    if valid_def:
        max_def_dist = max(
            math.hypot(p.x - SITE_A_ANCHOR[0], p.y - SITE_A_ANCHOR[1]) for p in valid_def
        )
        def_rot_sec = min(30.0, max_def_dist / PLAYER_RUN_SPEED_UNITS_PER_SEC)
    else:
        def_rot_sec = 30.0

    safe_time = float(time_remaining_sec) if _is_valid_coord(time_remaining_sec) else 45.0
    safe_time = max(0.0, min(100.0, safe_time))
    econ_diff_norm = max(-1.0, min(1.0, (atk_econ - def_econ) / 25000.0))

    return {
        "atk_alive": float(len(valid_atk)),
        "def_alive": float(len(valid_def)),
        "atk_control_ratio": round(atk_cells / total_cells, 6),
        "def_control_ratio": round(def_cells / total_cells, 6),
        "op_alive_def": op_alive_def,
        "op_exposed_area_ratio": round(op_exposed_cells / total_cells, 6),
        "op_targets_in_sight": float(op_targets),
        "atk_min_site_dist": round(atk_min_site_dist, 4),
        "def_rotation_time_sec": round(def_rot_sec, 4),
        "spike_planted": 1.0 if spike_planted else 0.0,
        "time_remaining_sec": round(safe_time, 4),
        "econ_diff_norm": round(econ_diff_norm, 6),
    }


def feature_dict_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict into a strict float32 vector matching FEATURE_ORDER."""
    return np.array([float(features[name]) for name in FEATURE_ORDER], dtype=np.float32)


def compute_operator_credit_risk(attacker_win_prob: float, op_alive_def: bool) -> float:
    """Expected credit loss if Defender Operator dies on round loss."""
    if not op_alive_def:
        return 0.0
    return round(float(attacker_win_prob) * OPERATOR_COST_CREDITS, 2)
