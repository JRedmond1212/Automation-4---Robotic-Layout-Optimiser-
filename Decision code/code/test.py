# test.py — Robotic Packing Line Config Optimiser
# ─────────────────────────────────────────────────────────────────────────────
# Changes in this version:
#   - No cap on configurations evaluated (all feasible combos run)
#   - File upload removed — always uses default module table
#   - Expanders for module/area/volume info removed
#   - Layout codes decoded to human-readable names in ranking table
#   - Throughput vs Cost scatter plots removed
#   - Per-robot detail table removed
#   - Aggregate plots gain secondary Y-axis (1000 L/hr) with median bar overlay
#   - Select Configuration moved to top of results (just below progress bar)
#   - Results ranking table wrapped in expander below aggregate plots
#   - Heatmap colour scale scoped to selected config range, not global
#
# Simulation engine:
#   Single-robot        -> analytical fast path (no overhead)
#   Multi-robot (scan)  -> pure heapq DES  (no SimPy generator overhead)
#   Multi-robot (detail)-> SimPy process-based DES
#
# Performance:
#   Adaptive coarse ratio grid per robot count.
#   Fine grid (0.02 step) always applied for the selected config heatmap.
#
# Requirements: pip install streamlit simpy numpy pandas plotly openpyxl
# -------------------------------------------------------------------------

import heapq
import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import simpy
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Tuning ────────────────────────────────────────────────────────────────────
FINE_STEP_FOR_SELECTED = 0.02
PLOT_HEIGHT            = 320
PROGRESS_UPDATE_SEC    = 0.2
FINE_HEATMAP_GRID_N    = 260

# Adaptive ratio grid: coarser for large N to manage combinatorial explosion.
# Fine grid always used for the selected config heatmap (never compromised).
# N<=3: 66 pts | N=4: 21 pts (~3x) | N=5: 15 pts (~4x) | N=6: 6 pts (~11x)
COARSE_STEP_BY_N = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.25, 6: 0.5}

# ── Fixed empirical unit areas keyed by robot count ───────────────────────────
UNIT_AREAS: Dict[int, float] = {
    0: 0.00, 1: 1.00, 2: 2.33, 3: 3.91,
    4: 4.91, 5: 5.61, 6: 6.82,
}

# ── Volume constants ──────────────────────────────────────────────────────────
# Drum: 55 L, 208 kg  →  density = 208/55 = 3.782 kg/L
# Box:  80 kg / 3.782 →  ≈ 21.15 L
LITRES_PER_DRUM = 55.0
LITRES_PER_BOX  = 80.0 / (208.0 / 55.0)

# ── Order definitions ─────────────────────────────────────────────────────────
ORDER_TYPES = {
    "Order 1 (Box12)":     {"boxes": 12, "drums": 0},
    "Order 2 (Drum4)":     {"boxes":  0, "drums": 4},
    "Order 3 (Mixed6B2D)": {"boxes":  6, "drums": 2},
}


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModuleType:
    config_id:         str
    name:              str
    can_box:           bool
    box_per_cycle:     int
    can_drum:          bool
    drum_per_cycle:    int
    cycle_time_s:      float
    toolchange_time_s: float
    cost:              float
    area:              float
    mode:              str


@dataclass
class RobotInstance:
    robot_id:    int
    mtype:       ModuleType
    last_tool:   Optional[str] = None
    next_free_t: float = 0.0
    busy_time:   float = 0.0
    tool_time:   float = 0.0
    box_time:    float = 0.0
    drum_time:   float = 0.0
    cycles:      int   = 0


@dataclass
class OrderState:
    order_type: str
    rem_boxes:  int
    rem_drums:  int


# ── Module table ──────────────────────────────────────────────────────────────
def default_modules() -> Dict[str, ModuleType]:
    rows = [
        ("C01.0", "Boxes only — 1/cycle",                True,  1, False, 0, 15,  0, 150_000, 4, "box_only"),
        ("C01.1", "Boxes only — 2/cycle",                True,  2, False, 0, 15,  0, 150_000, 4, "box_only"),
        ("C01.2", "Boxes only — 3/cycle",                True,  3, False, 0, 15,  0, 150_000, 4, "box_only"),
        ("C01.3", "Boxes only — 4/cycle",                True,  4, False, 0, 45,  0, 150_000, 4, "box_only"),
        ("C01.4", "Boxes only — 6/cycle",                True,  6, False, 0, 45,  0, 150_000, 4, "box_only"),
        ("C02.0", "Drums only — 1/cycle",                False, 0, True,  1, 15,  0, 150_000, 4, "drum_only"),
        ("C02.1", "Drums only — 2/cycle",                False, 0, True,  2, 45,  0, 300_000, 4, "drum_only"),
        ("C03.0", "Dual built-in — 3 boxes OR 1 drum",   True,  3, True,  1, 15,  0, 180_000, 4, "dual_built_in"),
        ("C03.1", "Dual built-in — 6 boxes OR 2 drums",  True,  6, True,  2, 45,  0, 330_000, 4, "dual_built_in"),
        ("C04.0", "Dual toolchange — 3 boxes OR 1 drum", True,  3, True,  1, 15, 45, 180_000, 4, "dual_toolchange"),
        ("C04.1", "Dual toolchange — 6 boxes OR 2 drums",True,  6, True,  2, 45, 45, 330_000, 4, "dual_toolchange"),
    ]
    out: Dict[str, ModuleType] = {}
    for r in rows:
        out[r[0]] = ModuleType(
            config_id=r[0], name=r[1],
            can_box=bool(r[2]),  box_per_cycle=int(r[3]),
            can_drum=bool(r[4]), drum_per_cycle=int(r[5]),
            cycle_time_s=float(r[6]), toolchange_time_s=float(r[7]),
            cost=float(r[8]), area=float(r[9]), mode=str(r[10]),
        )
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────
def stepped_ratios(step: float) -> List[Tuple[float, float, float]]:
    n = int(round(1.0 / step))
    return [(i/n, j/n, (n-i-j)/n) for i in range(n+1) for j in range(n+1-i)]


def layout_to_str(layout: Dict[str, int]) -> str:
    return ", ".join(f"{k}x{v}" for k, v in sorted(layout.items()) if v > 0)


def parse_layout_str(s: str) -> Dict[str, int]:
    if not s:
        return {}
    result = {}
    for part in s.split(","):
        part = part.strip()
        if part:
            k, v = part.split("x")
            result[k.strip()] = int(v.strip())
    return result


def decode_layout_str(layout_str: str, modules: Dict[str, ModuleType]) -> str:
    """Convert 'C03.0x1, C01.2x2' → 'Dual built-in 3/1  +  2× Boxes only 3/cycle'."""
    layout = parse_layout_str(layout_str)
    parts = []
    for mid, cnt in sorted(layout.items()):
        if cnt > 0 and mid in modules:
            name = modules[mid].name
            parts.append(f"{cnt}× {name}" if cnt > 1 else name)
    return "  +  ".join(parts)


def compute_layout_cost(layout: Dict[str, int], modules: Dict[str, ModuleType]) -> float:
    return sum(cnt * modules[mid].cost for mid, cnt in layout.items() if cnt > 0)


def layout_can_do_mixed(layout: Dict[str, int], modules: Dict[str, ModuleType]) -> bool:
    has_box = has_drum = False
    for mid, cnt in layout.items():
        if cnt <= 0:
            continue
        mt       = modules[mid]
        has_box  = has_box  or (mt.can_box  and mt.box_per_cycle  > 0)
        has_drum = has_drum or (mt.can_drum and mt.drum_per_cycle > 0)
    return has_box and has_drum


# ── Enumeration by robot count ────────────────────────────────────────────────
def enumerate_layouts_for_n_robots(
    modules: Dict[str, ModuleType], n_robots: int
) -> List[Dict[str, int]]:
    """
    All EE combinations WITH repetition for exactly n_robots robots.
    Uses itertools.combinations_with_replacement — no area budget, no cap.
    """
    if n_robots == 0:
        return []
    ids     = sorted(modules.keys())
    layouts = []
    for combo in itertools.combinations_with_replacement(ids, n_robots):
        layout: Dict[str, int] = {}
        for mid in combo:
            layout[mid] = layout.get(mid, 0) + 1
        layouts.append(layout)
    return layouts


# ── Seed + cached sequence generation ────────────────────────────────────────
def make_seed(p: Tuple[float, float, float], n_orders: int) -> int:
    return abs(hash((p, int(n_orders)))) % 2_000_000_000


@st.cache_data(max_entries=4000)
def generate_order_sequence(
    p: Tuple[float, float, float], n_orders: int, seed: int
) -> Tuple[str, ...]:
    p1, p2, _ = p
    rng = random.Random(seed)
    seq = []
    for _ in range(n_orders):
        r = rng.random()
        if r < p1:          seq.append("Order 1 (Box12)")
        elif r < p1 + p2:   seq.append("Order 2 (Drum4)")
        else:               seq.append("Order 3 (Mixed6B2D)")
    return tuple(seq)


# ── Robot factory + work selection ───────────────────────────────────────────
def _build_robots(layout: Dict[str, int], modules: Dict[str, ModuleType]) -> List[RobotInstance]:
    robots = []
    rid    = 0
    for mid, cnt in sorted(layout.items()):
        if cnt <= 0:
            continue
        mt = modules[mid]
        for _ in range(cnt):
            robots.append(RobotInstance(robot_id=rid, mtype=mt))
            rid += 1
    return robots


def _choose_action(
    robot: RobotInstance, orders: List[OrderState]
) -> Optional[Tuple[int, str, int]]:
    mt = robot.mtype

    def try_box(idx, o):
        if o.rem_boxes > 0 and mt.can_box and mt.box_per_cycle > 0:
            return idx, "box", min(mt.box_per_cycle, o.rem_boxes)
        return None

    def try_drum(idx, o):
        if o.rem_drums > 0 and mt.can_drum and mt.drum_per_cycle > 0:
            return idx, "drum", min(mt.drum_per_cycle, o.rem_drums)
        return None

    # Dual robots: prefer same tool to reduce toolchanges
    if mt.can_box and mt.can_drum and robot.last_tool in ("box", "drum"):
        for idx, o in enumerate(orders):
            act = try_box(idx, o) if robot.last_tool == "box" else try_drum(idx, o)
            if act:
                return act

    for idx, o in enumerate(orders):
        if mt.mode == "box_only":
            act = try_box(idx, o)
        elif mt.mode == "drum_only":
            act = try_drum(idx, o)
        else:
            cb = o.rem_boxes > 0 and mt.can_box  and mt.box_per_cycle  > 0
            cd = o.rem_drums > 0 and mt.can_drum and mt.drum_per_cycle > 0
            if not (cb or cd):
                continue
            if cb and not cd:
                act = try_box(idx, o)
            elif cd and not cb:
                act = try_drum(idx, o)
            else:
                act = (try_box(idx, o)
                       if o.rem_boxes / mt.box_per_cycle >= o.rem_drums / mt.drum_per_cycle
                       else try_drum(idx, o))
        if act:
            return act
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  FAST ANALYTICAL PATH — single robot, no SimPy overhead
# ─────────────────────────────────────────────────────────────────────────────
def _sim_single_fast(mt: ModuleType, order_seq: Tuple[str, ...]) -> Tuple[float, float]:
    t         = 0.0
    last_tool = ""
    total_L   = 0.0
    dual      = mt.can_box and mt.can_drum
    tc_s      = mt.toolchange_time_s if dual else 0.0

    for order_key in order_seq:
        req   = ORDER_TYPES[order_key]
        rem_b = req["boxes"]
        rem_d = req["drums"]
        while rem_b > 0 or rem_d > 0:
            if rem_b > 0 and rem_d > 0:
                tool = last_tool if last_tool in ("box", "drum") else "box"
            elif rem_b > 0:
                tool = "box"
            else:
                tool = "drum"
            if dual and last_tool and last_tool != tool:
                t += tc_s
            t        += mt.cycle_time_s
            last_tool = tool
            if tool == "box":
                qty     = mt.box_per_cycle if rem_b >= mt.box_per_cycle else rem_b
                rem_b  -= qty
                total_L += qty * LITRES_PER_BOX
            else:
                qty     = mt.drum_per_cycle if rem_d >= mt.drum_per_cycle else rem_d
                rem_d  -= qty
                total_L += qty * LITRES_PER_DRUM
    return t, total_L


# -------------------------------------------------------------------------
#  HEAPQ FAST PATH  --  multi-robot scan loop (no SimPy generator overhead)
#
#  Mathematically identical to the SimPy path but avoids:
#    - Python generator context switches  (~2-3x faster per simulation)
#    - SimPy event object allocation
#    - Generator frame save/restore
#  Used exclusively in simulate_fast() for the ranking scan.
#  SimPy is kept for simulate_detailed() (explicit KPI tracking if needed).
# -------------------------------------------------------------------------
def _sim_multi_fast_heapq(
    layout: Dict[str, int],
    modules: Dict[str, ModuleType],
    order_seq: Tuple[str, ...],
) -> Tuple[float, float]:
    robots = _build_robots(layout, modules)
    if not robots or not order_seq:
        return 0.0, 0.0

    orders = [
        OrderState(ot, ORDER_TYPES[ot]["boxes"], ORDER_TYPES[ot]["drums"])
        for ot in order_seq
    ]

    evq    = [(0.0, r.robot_id) for r in robots]
    heapq.heapify(evq)
    by_id  = {r.robot_id: r for r in robots}
    total_L = 0.0
    last_t  = 0.0

    def all_done():
        return not any(o.rem_boxes > 0 or o.rem_drums > 0 for o in orders)

    while evq:
        t, rid = heapq.heappop(evq)
        robot  = by_id[rid]
        robot.next_free_t = t

        if all_done():
            break

        act = _choose_action(robot, orders)

        if act is None:
            if not all_done():
                # Jump to the next strictly-future event — avoids same-timestamp
                # spin (which caused infinite loops) and avoids the full-cycle
                # overestimate of t+cycle_time_s (which caused unnecessary rechecks).
                # O(k) scan over the k-element heap is negligible for k<=6.
                next_t = next(
                    (e[0] for e in evq if e[0] > t),
                    t + robot.mtype.cycle_time_s,   # fallback if no future event
                )
                heapq.heappush(evq, (next_t, rid))
            continue

        idx, item_type, qty = act
        o  = orders[idx]
        mt = robot.mtype

        tc = 0.0
        if mt.can_box and mt.can_drum and mt.toolchange_time_s > 0:
            if robot.last_tool is not None and robot.last_tool != item_type:
                tc = mt.toolchange_time_s

        # Claim work before scheduling (atomic — single-threaded)
        if item_type == "box":
            o.rem_boxes -= qty
            total_L     += qty * LITRES_PER_BOX
        else:
            o.rem_drums -= qty
            total_L     += qty * LITRES_PER_DRUM

        robot.last_tool = item_type
        dt     = tc + mt.cycle_time_s
        last_t = max(last_t, t + dt)
        heapq.heappush(evq, (t + dt, rid))

    return last_t, total_L


# ─────────────────────────────────────────────────────────────────────────────
#  SIMPY ENGINE — multi-robot DES
# ─────────────────────────────────────────────────────────────────────────────
def _simpy_simulate(
    layout: Dict[str, int],
    modules: Dict[str, ModuleType],
    order_seq: Tuple[str, ...],
    detailed: bool = False,
) -> Tuple[float, float, List[dict]]:
    """
    SimPy process-based DES. Each robot is an independent generator.
    Work is claimed before the first yield (atomic in SimPy's single-threaded model).
    Idle robots yield one cycle-time then retry.
    makespan = last productive cycle completion, not env.now after run().
    """
    if not order_seq:
        return 0.0, 0.0, []
    robots = _build_robots(layout, modules)
    if not robots:
        return 0.0, 0.0, []

    env    = simpy.Environment()
    orders = [
        OrderState(ot, ORDER_TYPES[ot]["boxes"], ORDER_TYPES[ot]["drums"])
        for ot in order_seq
    ]
    total_L = [0.0]
    last_t  = [0.0]

    def all_done():
        return not any(o.rem_boxes > 0 or o.rem_drums > 0 for o in orders)

    def robot_proc(robot: RobotInstance):
        mt = robot.mtype
        while not all_done():
            act = _choose_action(robot, orders)
            if act is None:
                yield env.timeout(mt.cycle_time_s)
                continue
            idx, item_type, qty = act
            o  = orders[idx]
            tc = 0.0
            if mt.can_box and mt.can_drum and mt.toolchange_time_s > 0:
                if robot.last_tool is not None and robot.last_tool != item_type:
                    tc = mt.toolchange_time_s
            if item_type == "box":
                o.rem_boxes -= qty
                L = qty * LITRES_PER_BOX
            else:
                o.rem_drums -= qty
                L = qty * LITRES_PER_DRUM
            robot.last_tool = item_type
            if tc > 0.0:
                yield env.timeout(tc)
            yield env.timeout(mt.cycle_time_s)
            total_L[0]       += L
            last_t[0]         = max(last_t[0], env.now)
            robot.next_free_t = env.now
            if detailed:
                robot.busy_time += tc + mt.cycle_time_s
                robot.tool_time += tc
                robot.cycles    += 1
                if item_type == "box":
                    robot.box_time += mt.cycle_time_s
                else:
                    robot.drum_time += mt.cycle_time_s

    for r in robots:
        env.process(robot_proc(r))
    env.run()
    makespan = last_t[0]

    if not detailed:
        return makespan, total_L[0], []

    ms = makespan if makespan > 1e-12 else 1.0
    per_robot_rows = []
    for r in robots:
        per_robot_rows.append({
            "Robot":                   f"R{r.robot_id + 1}",
            "Module":                  r.mtype.config_id,
            "Average utilisation":     r.busy_time / ms,
            "Average idle time":       max(0.0, 1.0 - r.busy_time / ms),
            "Average re tooling time": r.tool_time / ms,
            "% time used for drums":   r.drum_time / ms,
            "% time used for boxes":   r.box_time  / ms,
            "Cycles":                  r.cycles,
        })
    return makespan, total_L[0], per_robot_rows


# ── Public dispatchers ────────────────────────────────────────────────────────
def simulate_fast(
    layout: Dict[str, int], modules: Dict[str, ModuleType], order_seq: Tuple[str, ...]
) -> Tuple[float, float]:
    """
    Fast scan path.
      1 robot  -> analytical (pure arithmetic, zero allocation)
      N robots -> heapq DES  (no SimPy generator overhead)
    """
    robots = _build_robots(layout, modules)
    if not robots or not order_seq:
        return 0.0, 0.0
    if len(robots) == 1:
        return _sim_single_fast(robots[0].mtype, order_seq)
    return _sim_multi_fast_heapq(layout, modules, order_seq)


# ── Throughput helpers ────────────────────────────────────────────────────────
def thr_orders_hr(n_orders: int, makespan_s: float) -> float:
    return (n_orders / makespan_s) * 3600.0 if makespan_s > 1e-9 else 0.0


def thr_litres_hr(total_L: float, makespan_s: float) -> float:
    return (total_L / makespan_s) * 3600.0 if makespan_s > 1e-9 else 0.0


# ── Progress UI ───────────────────────────────────────────────────────────────
class ProgressUI:
    def __init__(self):
        self.task = st.empty()
        self.meta = st.empty()
        self.bar  = st.progress(0.0)
        self.t0   = time.perf_counter()
        self.last = self.t0

    @staticmethod
    def _fmt(s: float) -> str:
        if not math.isfinite(s) or s < 0: return "-"
        if s < 60:   return f"{s:.1f}s"
        if s < 3600: return f"{s/60:.1f} min"
        return f"{s/3600:.2f} hr"

    def update(self, name: str, done: int, total: int):
        now = time.perf_counter()
        if (now - self.last) < PROGRESS_UPDATE_SEC and done < total:
            return
        self.last = now
        elapsed   = now - self.t0
        rate      = done / elapsed if elapsed > 1e-9 else 0.0
        remaining = (total - done) / rate if rate > 1e-9 else float("inf")
        frac      = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))
        self.task.markdown(f"**Stage:** {name}")
        self.meta.markdown(
            f"**Runtime:** {self._fmt(elapsed)}  |  "
            f"**Est remaining:** {self._fmt(remaining)}  |  "
            f"**Tasks:** {done:,}/{total:,}"
        )
        self.bar.progress(frac)

    def finish(self): self.update("Done", 1, 1)


# ── Aggregate summary plots (dual Y-axis + 1000 L/hr bars) ───────────────────
def make_agg_plot(
    df_agg: pd.DataFrame,
    title:  str,
    y_max:  str,
    y_med:  str,
    y_min:  str,
    y_label: str,
    lph_col: str,
    lph_label: str,
    best_pts: list = None,
    sel_pt:   tuple = None,
) -> go.Figure:
    x   = df_agg["n_robots"].values
    fig = go.Figure()

    # Lines: max (green), median (dark), min (red) — no legend (shared caption)
    fig.add_trace(go.Scatter(
        x=x, y=df_agg[y_max], mode="lines+markers", name="Max",
        line=dict(dash="dot", width=1.5, color="#2ca02c"),
        marker=dict(size=4, color="#2ca02c"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_agg[y_med], mode="lines+markers", name="Median",
        line=dict(width=2.5, color="#085041"),
        marker=dict(size=5, color="#085041"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_agg[y_min], mode="lines+markers", name="Min",
        line=dict(dash="dot", width=1.5, color="#d62728"),
        marker=dict(size=4, color="#d62728"), showlegend=False,
    ))

    # 1000 L/hr bars on secondary (right) axis
    lph_vals = df_agg[lph_col].values / 1000.0
    fig.add_trace(go.Bar(
        x=x, y=lph_vals,
        name=lph_label, yaxis="y2",
        marker_color="rgba(186,117,23,0.35)",
        marker_line_color="rgba(186,117,23,0.7)",
        marker_line_width=1, width=0.4, showlegend=False,
    ))

    # Best config per N (red dots)
    if best_pts:
        bx = [p[0] for p in best_pts]
        by = [p[1] for p in best_pts]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="markers", name="Best per N",
            marker=dict(size=7, color="red", symbol="circle",
                        line=dict(color="darkred", width=1.2)),
            hovertemplate="Best config N=%{x}<br>%{y:.2f}<extra></extra>",
            showlegend=False,
        ))

    # Selected config (blue dot)
    if sel_pt is not None:
        fig.add_trace(go.Scatter(
            x=[sel_pt[0]], y=[sel_pt[1]], mode="markers", name="Selected",
            marker=dict(size=9, color="royalblue", symbol="circle",
                        line=dict(color="navy", width=1.2)),
            hovertemplate="Selected N=%{x}<br>%{y:.2f}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=title, height=PLOT_HEIGHT,
        dragmode=False, hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=60, t=50, b=40),
        yaxis=dict(title=y_label, rangemode="tozero", side="left"),
        yaxis2=dict(title=lph_label, rangemode="tozero",
                    overlaying="y", side="right", showgrid=False),
        barmode="overlay",
    )
    fig.update_xaxes(title_text="Number of robots", tickmode="linear", tick0=0, dtick=1)
    return fig


# ── Triangle heatmap ──────────────────────────────────────────────────────────
def bary_to_xy(p1, p2, p3):
    return p2 * 1.0 + p3 * 0.5, p3 * (math.sqrt(3) / 2.0)


def inside_triangle_mask(X, Y):
    h = math.sqrt(3) / 2.0
    m = math.sqrt(3)
    return (
        (Y >= -1e-12) & (Y <= m * X + 1e-12) &
        (Y <= m * (1.0 - X) + 1e-12) & (Y <= h + 1e-12) &
        (X >= -1e-12) & (X <= 1.0 + 1e-12)
    )


def idw_interpolate(xs, ys, zs, gx, gy, power=2.0):
    dx = gx.reshape(-1, 1) - xs.reshape(1, -1)
    dy = gy.reshape(-1, 1) - ys.reshape(1, -1)
    w  = 1.0 / (dx * dx + dy * dy + 1e-9) ** (power / 2.0)
    return ((w @ zs.reshape(-1, 1)) / w.sum(axis=1, keepdims=True)).reshape(gx.shape)


def make_triangle_heatmap(points_df, title, zmin, zmax):
    p1 = points_df["p1"].to_numpy(float)
    p2 = points_df["p2"].to_numpy(float)
    p3 = points_df["p3"].to_numpy(float)
    z  = points_df["throughput"].to_numpy(float)
    xs, ys = bary_to_xy(p1, p2, p3)
    n     = FINE_HEATMAP_GRID_N
    h     = math.sqrt(3) / 2.0
    x_lin = np.linspace(0.0, 1.0, n)
    y_lin = np.linspace(0.0, h, n)
    X, Y  = np.meshgrid(x_lin, y_lin)
    mask  = inside_triangle_mask(X, Y)
    Z     = idw_interpolate(xs, ys, z, X.ravel(), Y.ravel()).reshape(X.shape)
    Z_m   = np.where(mask, Z, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_lin, y=y_lin, z=Z_m,
        colorscale=[[0.0, "#67000d"], [0.5, "#fdae61"], [1.0, "#d9f0a3"]],
        zmin=zmin, zmax=zmax,
        colorbar=dict(title="Orders/hr"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=5, color="rgba(0,0,0,0.25)"),
        customdata=np.stack([p1, p2, p3, z], axis=1),
        hovertemplate=(
            "Order 1: %{customdata[0]:.2f}<br>Order 2: %{customdata[1]:.2f}<br>"
            "Order 3: %{customdata[2]:.2f}<br>Throughput: %{customdata[3]:.2f} orders/hr<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0, 0.5, 0.0], y=[0.0, 0.0, h, 0.0],
        mode="lines", line=dict(color="black"), showlegend=False,
    ))
    for x_, y_, txt, xa, ya in [
        (0.0, 0.0, "Order 1", "left",   "top"),
        (1.0, 0.0, "Order 2", "right",  "top"),
        (0.5,  h,  "Order 3", "center", "bottom"),
    ]:
        fig.add_annotation(x=x_, y=y_, text=txt, showarrow=False, xanchor=xa, yanchor=ya)
    fig.update_layout(
        title=title, height=440, dragmode=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(visible=False, range=[-0.02, 1.02])
    fig.update_yaxes(visible=False, range=[-0.02, h + 0.04], scaleanchor="x", scaleratio=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Packing Line Optimiser", layout="wide")
st.title("Automated Packing Line: Configuration & Robot Selection Tool")


with st.sidebar:
    st.header("Inputs")
    with st.form("form", clear_on_submit=False):
        max_robots = st.slider(
            "Max number of robots",
            min_value=0, max_value=6, value=4, step=1,
            help="Evaluates every EE combination for N robots.",
        )
        n_orders = st.slider(
            "Orders per Monte Carlo Sim",
            min_value=20, max_value=400, value=200, step=10,
            help="Number of orders queued per mixture ratio.",
        )
        run = st.form_submit_button("Run")

# Always use default modules
modules = default_modules()

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("results", "agg_stats"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Run ───────────────────────────────────────────────────────────────────────
if run:
    try:
        prog = ProgressUI()

        if max_robots == 0:
            st.session_state.results   = pd.DataFrame()
            st.session_state.agg_stats = pd.DataFrame()
            prog.finish()
        else:
            # Pre-generate sequences per robot count (different coarse step per N).
            # Sequences are cached so overlapping ratio points are reused.
            seqs_by_n: Dict[int, Dict[Tuple, Tuple[str, ...]]] = {}
            all_ratios_per_n: Dict[int, List] = {}
            for n in range(1, max_robots + 1):
                step = COARSE_STEP_BY_N.get(n, 0.1)
                ratios = stepped_ratios(step)
                all_ratios_per_n[n] = ratios
                seqs: Dict[Tuple, Tuple[str, ...]] = {}
                for p in ratios:
                    seed     = make_seed(p, n_orders)
                    seqs[p]  = generate_order_sequence(p, n_orders, seed)
                seqs_by_n[n] = seqs

            # Enumerate all feasible layouts per N — no cap
            prog.update("Enumerating layouts per robot count...", 0, max_robots)
            all_feasible: Dict[int, List] = {}
            for n in range(1, max_robots + 1):
                layouts_n  = enumerate_layouts_for_n_robots(modules, n)
                feasible_n = [lay for lay in layouts_n if layout_can_do_mixed(lay, modules)]
                all_feasible[n] = feasible_n
                prog.update("Enumerating layouts per robot count...", n, max_robots)

            total_tasks = sum(len(all_feasible[n]) * len(all_ratios_per_n[n]) for n in range(1, max_robots + 1))
            if total_tasks == 0:
                st.session_state.results   = pd.DataFrame()
                st.session_state.agg_stats = pd.DataFrame()
                prog.finish()
            else:
                all_rows:       List[dict] = []
                agg_stats_rows: List[dict] = []
                done = 0

                # N=0 baseline row (no robots = zero output)
                agg_stats_rows.append({
                    "n_robots": 0, "unit_area": 0.0,
                    "max_thr": 0.0, "median_thr": 0.0, "min_thr": 0.0,
                    "max_tpa": 0.0, "median_tpa": 0.0, "min_tpa": 0.0,
                    "median_lph": 0.0, "median_lph_per_area": 0.0,
                })

                for n in range(1, max_robots + 1):
                    unit_area  = UNIT_AREAS[n]
                    feasible_n = all_feasible[n]

                    if not feasible_n:
                        agg_stats_rows.append({
                            "n_robots": n, "unit_area": unit_area,
                            "max_thr": 0.0, "median_thr": 0.0, "min_thr": 0.0,
                            "max_tpa": 0.0, "median_tpa": 0.0, "min_tpa": 0.0,
                            "median_lph": 0.0, "median_lph_per_area": 0.0,
                        })
                        continue

                    combo_med_thrs: List[float] = []
                    combo_med_tpas: List[float] = []
                    combo_med_lphs: List[float] = []

                    for lay in feasible_n:
                        ls   = layout_to_str(lay)
                        cost = compute_layout_cost(lay, modules)
                        thr_list: List[float] = []
                        lph_list: List[float] = []

                        for p, seq in seqs_by_n[n].items():
                            makespan, total_L = simulate_fast(lay, modules, seq)
                            thr = thr_orders_hr(len(seq), makespan)
                            lph = thr_litres_hr(total_L, makespan)
                            thr_list.append(thr)
                            lph_list.append(lph)
                            done += 1
                            prog.update(
                                f"N={n} robots — {len(feasible_n)} combos...",
                                done, total_tasks
                            )

                        thr_arr  = np.array(thr_list, float)
                        lph_arr  = np.array(lph_list, float)
                        med_thr  = float(np.median(thr_arr))
                        med_lph  = float(np.median(lph_arr))
                        med_tpa  = med_thr / unit_area if unit_area > 0 else 0.0
                        med_lpa  = med_lph / unit_area if unit_area > 0 else 0.0

                        combo_med_thrs.append(med_thr)
                        combo_med_tpas.append(med_tpa)
                        combo_med_lphs.append(med_lph)

                        med_lph_per_ua = med_lph / unit_area if unit_area > 0 else 0.0
                        all_rows.append({
                            "layout_str":                      ls,
                            "config_st_name":                  ls,
                            "n_robots":                        n,
                            "unit_area":                       unit_area,
                            "cost":                            float(cost),
                            "max_throughput":                  float(np.max(thr_arr)),
                            "min_throughput":                  float(np.min(thr_arr)),
                            "median_throughput":               med_thr,
                            "median_litres_per_hr":            med_lph,
                            "median_throughput_per_unit_area": med_tpa,
                            "median_lph_per_unit_area":        med_lph_per_ua,
                        })

                    agg_stats_rows.append({
                        "n_robots":            n,
                        "unit_area":           unit_area,
                        "max_thr":             float(np.max(combo_med_thrs)),
                        "median_thr":          float(np.median(combo_med_thrs)),
                        "min_thr":             float(np.min(combo_med_thrs)),
                        "max_tpa":             float(np.max(combo_med_tpas)),
                        "median_tpa":          float(np.median(combo_med_tpas)),
                        "min_tpa":             float(np.min(combo_med_tpas)),
                        "median_lph":          float(np.median(combo_med_lphs)),
                        "median_lph_per_area": float(np.median(
                            [l / unit_area if unit_area > 0 else 0.0 for l in combo_med_lphs]
                        )),
                    })

                df_out = (
                    pd.DataFrame(all_rows)
                    .sort_values("median_lph_per_unit_area", ascending=False)
                    .reset_index(drop=True)
                )
                df_agg = pd.DataFrame(agg_stats_rows)

                st.session_state.results   = df_out
                st.session_state.agg_stats = df_agg
                prog.finish()

    except Exception as e:
        st.session_state.results   = None
        st.session_state.agg_stats = None
        st.error("Run failed. Details below:")
        st.exception(e)

# ── Guard ─────────────────────────────────────────────────────────────────────
df     = st.session_state.results
df_agg = st.session_state.agg_stats

if df is None:
    st.info("Set inputs and press **Run**.")
    st.stop()

if df.empty or df_agg is None or df_agg.empty:
    st.warning("No feasible mixed-capable layouts found. Try increasing the max robot count.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — Select Configuration first (top of page)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Select Configuration")
df["_label"] = df.apply(
    lambda r: (
        f"[{int(r['n_robots'])}R | {r['median_lph_per_unit_area']/1000:.1f} kL/hr/area]  "
        f"{decode_layout_str(r['layout_str'], modules)}"
    ),
    axis=1,
)
selected_idx = st.selectbox(
    "Configuration (N robots | kL/hr/unit area | EE Configuration)",
    range(len(df)),
    format_func=lambda i: df["_label"].iloc[i],
    index=0,
)
selected_row  = df.iloc[selected_idx]
selected      = selected_row["layout_str"]
sel_n_robots  = int(selected_row["n_robots"])
sel_unit_area = selected_row["unit_area"]

st.caption(
    f"**{sel_n_robots} robot{'s' if sel_n_robots != 1 else ''}** — "
    f"unit area = {sel_unit_area:.2f}  |  "
    f"cost = £{int(selected_row['cost']):,}  |  "
    f"median throughput = {selected_row['median_throughput']:.1f} orders/hr  |  "
    f"median L/hr = {selected_row['median_litres_per_hr']:,.0f} L/hr  |  "
    f"median kL/hr/unit area = {selected_row['median_lph_per_unit_area']/1000:.2f}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary plots
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Throughput vs Number of Robots")


# Best config per N (by median L/hr per unit area) and selected config coords
ns_evaluated = sorted(df["n_robots"].unique())
best_by_n = {
    n: df[df["n_robots"] == n].loc[df[df["n_robots"] == n]["median_lph_per_unit_area"].idxmax()]
    for n in ns_evaluated
}
best_pts_p1 = [(n, row["median_throughput"]) for n, row in best_by_n.items()]
best_pts_p2 = [(n, row["median_throughput_per_unit_area"]) for n, row in best_by_n.items()]
sel_pt_p1   = (sel_n_robots, float(selected_row["median_throughput"]))
sel_pt_p2   = (sel_n_robots, float(selected_row["median_throughput_per_unit_area"]))

c1, c2 = st.columns(2, gap="small")
with c1:
    st.plotly_chart(
        make_agg_plot(
            df_agg,
            title="Orders/hr vs Number of Robots",
            y_max="max_thr", y_med="median_thr", y_min="min_thr",
            y_label="Orders / hr",
            lph_col="median_lph",
            lph_label="Median 1000 L/hr",
            best_pts=best_pts_p1,
            sel_pt=sel_pt_p1,
        ),
        use_container_width=True, config={"displayModeBar": False},
    )
with c2:
    st.plotly_chart(
        make_agg_plot(
            df_agg,
            title="Orders/hr per Unit Area vs Number of Robots",
            y_max="max_tpa", y_med="median_tpa", y_min="min_tpa",
            y_label="Orders / hr / unit area",
            lph_col="median_lph_per_area",
            lph_label="Median 1000 L/hr / unit area",
            best_pts=best_pts_p2,
            sel_pt=sel_pt_p2,
        ),
        use_container_width=True, config={"displayModeBar": False},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Results Ranking Table — collapsible expander
# ─────────────────────────────────────────────────────────────────────────────
with st.expander(
    f"Results Ranking Table — {len(df):,} configurations, ranked by median kL/hr/unit area",
    expanded=False,
):
    st.caption("Ranked by median L/hr per unit area (best throughput efficeny).")

    display_df = df[[
        "n_robots", "layout_str", "unit_area", "cost",
        "median_lph_per_unit_area", "median_litres_per_hr",
        "median_throughput", "max_throughput", "min_throughput",
    ]].copy()
    display_df.insert(
        2, "configuration",
        df["layout_str"].apply(lambda s: decode_layout_str(s, modules))
    )
    display_df = display_df.drop(columns=["layout_str"])
    # Convert L/hr columns to kL/hr for display
    display_df["median_lph_per_unit_area"] = display_df["median_lph_per_unit_area"] / 1000.0
    display_df["median_litres_per_hr"]     = display_df["median_litres_per_hr"]     / 1000.0
    display_df.columns = [
        "robots", "configuration", "unit area", "cost (£)",
        "median kL/hr/area", "median kL/hr",
        "median thr (orders/hr)", "max thr (orders/hr)", "min thr (orders/hr)",
    ]
    st.dataframe(display_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Order Mixture Heatmap — colour scale scoped to selected config only
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Order Mixture Ratio Heatmap (Shows Best and Worst Case Scenarios)")
ratios_fine = stepped_ratios(FINE_STEP_FOR_SELECTED)
sel_layout  = parse_layout_str(selected)
sel_points  = []

for p in ratios_fine:
    seed     = make_seed(p, n_orders)
    seq      = generate_order_sequence(p, n_orders, seed)
    makespan, total_L = simulate_fast(sel_layout, modules, seq)
    sel_points.append({
        "p1": p[0], "p2": p[1], "p3": p[2],
        "throughput": thr_orders_hr(len(seq), makespan),
    })

# Scale spans the min–max of ALL configs for the same robot count,
# so colour is comparable across configs at that tier.
# Uses coarse-grid min_throughput/max_throughput already in df.
df_n        = df[df["n_robots"] == sel_n_robots]
heatmap_min = float(df_n["min_throughput"].min())
heatmap_max = float(df_n["max_throughput"].max())
if abs(heatmap_max - heatmap_min) < 0.5:
    heatmap_min = max(0.0, heatmap_max - 1.0)

st.plotly_chart(
    make_triangle_heatmap(
        pd.DataFrame(sel_points),
        title=(
            f"Order-ratio heatmap — "
            f"{decode_layout_str(selected, modules)}  "
            f"({sel_n_robots}R, unit area {sel_unit_area})"
        ),
        zmin=heatmap_min,
        zmax=heatmap_max,
    ),
    use_container_width=True,
    config={"displayModeBar": False},
)
