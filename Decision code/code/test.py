# test.py — Single-file Streamlit app (Cloud-safe, NO st.cache_data)
# -----------------------------------------------------------------------------------------
# What this version fixes / improves:
# 1) Removes all Streamlit caching decorators (as requested)
# 2) Replaces deprecated use_container_width -> width="stretch"
# 3) Forces Excel engine="openpyxl" for robust xlsx reading
# 4) Adds safety limits to avoid Streamlit Cloud hard-kills (OOM/CPU):
#    - caps enumerated layouts
#    - caps evaluated layouts (sample top-N)
# 5) Adds clearer error handling and progress updates

import math
import random
import time
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# Fixed knobs (keep these stable)
# -----------------------------
COARSE_STEP = 0.1
FINE_STEP_FOR_SELECTED = 0.02
PLOT_HEIGHT = 260
PROGRESS_UPDATE_SEC = 0.2

# Cloud-safety caps (IMPORTANT)
MAX_ENUMERATED_LAYOUTS = 5000        # cap how many layouts recursion can generate
MAX_EVALUATED_LAYOUTS = 400          # cap how many layouts we actually simulate
FINE_HEATMAP_GRID_N = 260            # lower than 360 to reduce CPU/RAM

# Litres proxy assumption:
LITRES_PER_DRUM = 208.0


# -----------------------------
# Orders (requirements per pallet)
# -----------------------------
ORDER_TYPES = {
    "Order 1 (Box12)": {"boxes": 12, "drums": 0},
    "Order 2 (Drum4)": {"boxes": 0, "drums": 4},
    "Order 3 (Mixed6B2D)": {"boxes": 6, "drums": 2},
}
ORDER_KEYS = list(ORDER_TYPES.keys())


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class ModuleType:
    config_id: str
    name: str
    can_box: bool
    box_per_cycle: int
    can_drum: bool
    drum_per_cycle: int
    cycle_time_s: float
    toolchange_time_s: float
    cost: float
    area: float
    mode: str  # "box_only", "drum_only", "dual_built_in", "dual_toolchange"


@dataclass
class RobotInstance:
    robot_id: int
    mtype: ModuleType
    last_tool: Optional[str] = None  # "box" or "drum"
    next_free_t: float = 0.0

    busy_time: float = 0.0
    tool_time: float = 0.0
    box_time: float = 0.0
    drum_time: float = 0.0
    cycles: int = 0


@dataclass
class OrderState:
    order_type: str
    rem_boxes: int
    rem_drums: int


# -----------------------------
# Default module table
# -----------------------------
def default_modules() -> Dict[str, ModuleType]:
    rows = [
        ("C01.0", "boxes only, x1", True, 1, False, 0, 10, 0, 10000, 4, "box_only"),
        ("C01.1", "boxes only, x2", True, 2, False, 0, 10, 0, 20000, 4, "box_only"),
        ("C01.2", "boxes only, x3", True, 3, False, 0, 10, 0, 30000, 4, "box_only"),
        ("C01.3", "boxes only, x4", True, 4, False, 0, 30, 0, 40000, 4, "box_only"),
        ("C01.4", "boxes only, x6", True, 6, False, 0, 30, 0, 150000, 4, "box_only"),
        ("C02.0", "barrels only, x1", False, 0, True, 1, 10, 0, 30000, 4, "drum_only"),
        ("C02.1", "barrels only, x2", False, 0, True, 2, 10, 0, 150000, 4, "drum_only"),

        # dual robots do ONE product type per cycle (either boxes OR drums)
        ("C03.0", "boxes and barrels (Built in), x3 OR x1", True, 3, True, 1, 10, 10, 30000, 4, "dual_built_in"),
        ("C03.1", "boxes and barrels (Built in), x6 OR x2", True, 6, True, 2, 30, 10, 150000, 4, "dual_built_in"),
        ("C04.0", "boxes and barrels (Tool change), x3 OR x1", True, 3, True, 1, 10, 30, 30000, 4, "dual_toolchange"),
        ("C04.1", "boxes and barrels (Tool change), x6 OR x2", True, 6, True, 2, 30, 30, 150000, 4, "dual_toolchange"),
    ]
    out: Dict[str, ModuleType] = {}
    for r in rows:
        out[r[0]] = ModuleType(
            config_id=r[0],
            name=r[1],
            can_box=bool(r[2]),
            box_per_cycle=int(r[3]),
            can_drum=bool(r[4]),
            drum_per_cycle=int(r[5]),
            cycle_time_s=float(r[6]),
            toolchange_time_s=float(r[7]),
            cost=float(r[8]),
            area=float(r[9]),
            mode=str(r[10]),
        )
    return out


def load_modules_from_excel_bytes(uploaded_bytes: bytes) -> Dict[str, ModuleType]:
    # Force engine for cloud consistency
    df = pd.read_excel(uploaded_bytes, engine="openpyxl")

    def find_col(cands):
        for c in cands:
            for col in df.columns:
                if str(col).strip().lower() == c.strip().lower():
                    return col
        return None

    col_id = find_col(["Config ID", "ConfigID", "ID"])
    col_name = find_col(["Config name", "Config", "Name"])
    col_b_ok = find_col(["Boxes?", "Boxes"])
    col_b = find_col(["Boxes per cycle", "Box per cycle", "B"])
    col_d_ok = find_col(["Barrels?", "Drums?", "Barrels", "Drums"])
    col_d = find_col(["Barrels per cycle", "Drums per cycle", "D"])
    col_tc = find_col(["Cycle time", "Cycle Time", "t_cycle", "t"])
    col_tool = find_col(["Toolchange time", "Tool change", "Toolchange", "t_tool"])
    col_cost = find_col(["Cost", "Cost (£)", "£"])
    col_area = find_col(["Area", "Footprint", "m2"])

    required = [col_id, col_name, col_tc, col_tool, col_cost, col_area]
    if any(c is None for c in required):
        raise ValueError(f"Excel columns missing/unrecognized. Found: {list(df.columns)}")

    out: Dict[str, ModuleType] = {}
    for _, row in df.iterrows():
        cid = str(row[col_id]).strip()
        name = str(row[col_name]).strip()

        can_box = str(row[col_b_ok]).strip().lower().startswith("y") if col_b_ok else (float(row[col_b]) > 0)
        can_drum = str(row[col_d_ok]).strip().lower().startswith("y") if col_d_ok else (float(row[col_d]) > 0)
        bpc = int(row[col_b]) if col_b is not None else 0
        dpc = int(row[col_d]) if col_d is not None else 0

        t_cycle = float(row[col_tc])
        t_tool = float(row[col_tool])
        cost = float(row[col_cost])
        area = float(row[col_area])

        if can_box and not can_drum:
            mode = "box_only"
        elif can_drum and not can_box:
            mode = "drum_only"
        else:
            mode = "dual_toolchange" if t_tool > 0 else "dual_built_in"

        out[cid] = ModuleType(
            config_id=cid,
            name=name,
            can_box=bool(can_box),
            box_per_cycle=int(bpc),
            can_drum=bool(can_drum),
            drum_per_cycle=int(dpc),
            cycle_time_s=float(t_cycle),
            toolchange_time_s=float(t_tool),
            cost=float(cost),
            area=float(area),
            mode=str(mode),
        )
    return out


# -----------------------------
# Helpers
# -----------------------------
def stepped_ratios(step: float) -> List[Tuple[float, float, float]]:
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i / n, j / n, k / n))
    return out


def layout_to_str(layout: Dict[str, int]) -> str:
    return ", ".join([f"{k}×{v}" for k, v in sorted(layout.items()) if v > 0])


def parse_layout_str(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        k, v = p.split("×")
        out[k.strip()] = int(v.strip())
    return out


def compute_layout_cost_area(layout: Dict[str, int], modules: Dict[str, ModuleType]) -> Tuple[float, float]:
    cost = 0.0
    area = 0.0
    for mid, cnt in layout.items():
        if cnt <= 0:
            continue
        mt = modules[mid]
        cost += cnt * mt.cost
        area += cnt * mt.area
    return cost, area


def layout_can_do_mixed(layout: Dict[str, int], modules: Dict[str, ModuleType]) -> bool:
    has_box = False
    has_drum = False
    for mid, cnt in layout.items():
        if cnt <= 0:
            continue
        mt = modules[mid]
        has_box = has_box or (mt.can_box and mt.box_per_cycle > 0)
        has_drum = has_drum or (mt.can_drum and mt.drum_per_cycle > 0)
    return has_box and has_drum


# -----------------------------
# Layout enumeration (NO cache, capped)
# -----------------------------
def enumerate_layouts(modules: Dict[str, ModuleType], max_area: float) -> List[Dict[str, int]]:
    ids = sorted(modules.keys(), key=lambda k: (modules[k].area, k))
    cur: Dict[str, int] = {}
    layouts: List[Dict[str, int]] = []

    def rec(i: int, used_area: float):
        if len(layouts) >= MAX_ENUMERATED_LAYOUTS:
            return
        if used_area > max_area + 1e-9:
            return
        if i == len(ids):
            if sum(cur.values()) >= 1:
                layouts.append(dict(cur))
            return

        mid = ids[i]
        a = modules[mid].area
        max_cnt = int((max_area - used_area) // a) if a > 0 else 0

        # iterate 0..max_cnt
        for c in range(max_cnt + 1):
            if len(layouts) >= MAX_ENUMERATED_LAYOUTS:
                return
            if c == 0:
                cur.pop(mid, None)
                rec(i + 1, used_area)
            else:
                cur[mid] = c
                rec(i + 1, used_area + c * a)

        cur.pop(mid, None)

    rec(0, 0.0)
    return layouts


# -----------------------------
# Random order sequence (NO cache)
# -----------------------------
def generate_order_sequence(p: Tuple[float, float, float], n_orders: int, seed: int) -> List[str]:
    p1, p2, p3 = p
    rng = random.Random(seed)
    seq = []
    for _ in range(n_orders):
        r = rng.random()
        if r < p1:
            seq.append("Order 1 (Box12)")
        elif r < p1 + p2:
            seq.append("Order 2 (Drum4)")
        else:
            seq.append("Order 3 (Mixed6B2D)")
    return seq


# -----------------------------
# Simple agent-like simulation for ONE (layout, sequence)
# -----------------------------
def build_robot_instances(layout: Dict[str, int], modules: Dict[str, ModuleType]) -> List[RobotInstance]:
    robots: List[RobotInstance] = []
    rid = 0
    for mid, cnt in sorted(layout.items()):
        if cnt <= 0:
            continue
        mt = modules[mid]
        for _ in range(cnt):
            robots.append(RobotInstance(robot_id=rid, mtype=mt))
            rid += 1
    return robots


def robot_can_do(robot: RobotInstance, item_type: str) -> bool:
    mt = robot.mtype
    if item_type == "box":
        return mt.can_box and mt.box_per_cycle > 0
    return mt.can_drum and mt.drum_per_cycle > 0


def choose_action(robot: RobotInstance, orders: List[OrderState]) -> Optional[Tuple[int, str, int]]:
    mt = robot.mtype

    def needs_box(o: OrderState) -> bool:
        return o.rem_boxes > 0

    def needs_drum(o: OrderState) -> bool:
        return o.rem_drums > 0

    # Prefer same tool for dual-capable robots
    if mt.can_box and mt.can_drum and robot.last_tool in ("box", "drum"):
        t = robot.last_tool
        for idx, o in enumerate(orders):
            if t == "box" and needs_box(o) and robot_can_do(robot, "box"):
                qty = min(mt.box_per_cycle, o.rem_boxes)
                return idx, "box", qty
            if t == "drum" and needs_drum(o) and robot_can_do(robot, "drum"):
                qty = min(mt.drum_per_cycle, o.rem_drums)
                return idx, "drum", qty

    # Earliest compatible
    for idx, o in enumerate(orders):
        if mt.mode == "box_only":
            if needs_box(o) and robot_can_do(robot, "box"):
                qty = min(mt.box_per_cycle, o.rem_boxes)
                return idx, "box", qty
        elif mt.mode == "drum_only":
            if needs_drum(o) and robot_can_do(robot, "drum"):
                qty = min(mt.drum_per_cycle, o.rem_drums)
                return idx, "drum", qty
        else:
            can_b = needs_box(o) and robot_can_do(robot, "box")
            can_d = needs_drum(o) and robot_can_do(robot, "drum")
            if not (can_b or can_d):
                continue

            if can_b and not can_d:
                qty = min(mt.box_per_cycle, o.rem_boxes)
                return idx, "box", qty
            if can_d and not can_b:
                qty = min(mt.drum_per_cycle, o.rem_drums)
                return idx, "drum", qty

            score_b = o.rem_boxes / max(1, mt.box_per_cycle)
            score_d = o.rem_drums / max(1, mt.drum_per_cycle)
            if score_b >= score_d:
                qty = min(mt.box_per_cycle, o.rem_boxes)
                return idx, "box", qty
            else:
                qty = min(mt.drum_per_cycle, o.rem_drums)
                return idx, "drum", qty

    return None


def simulate_sequence(
    layout: Dict[str, int],
    modules: Dict[str, ModuleType],
    order_seq: List[str],
) -> Tuple[float, float, List[dict]]:
    orders: List[OrderState] = []
    for ot in order_seq:
        req = ORDER_TYPES[ot]
        orders.append(OrderState(order_type=ot, rem_boxes=req["boxes"], rem_drums=req["drums"]))

    robots = build_robot_instances(layout, modules)
    if not robots or not orders:
        return 0.0, 0.0, []

    evq = [(0.0, r.robot_id) for r in robots]
    heapq.heapify(evq)
    robots_by_id = {r.robot_id: r for r in robots}

    total_drums_done = 0.0

    while evq:
        t, rid = heapq.heappop(evq)
        robot = robots_by_id[rid]
        robot.next_free_t = t

        if not any((o.rem_boxes > 0 or o.rem_drums > 0) for o in orders):
            break

        act = choose_action(robot, orders)
        if act is None:
            heapq.heappush(evq, (t + 1e9, rid))
            continue

        idx, item_type, qty = act
        mt = robot.mtype
        o = orders[idx]

        tool_t = 0.0
        if mt.can_box and mt.can_drum and mt.toolchange_time_s > 0:
            if robot.last_tool is not None and robot.last_tool != item_type:
                tool_t = mt.toolchange_time_s

        cycle_t = mt.cycle_time_s
        dt = tool_t + cycle_t

        if item_type == "box":
            o.rem_boxes -= qty
            robot.box_time += cycle_t
        else:
            o.rem_drums -= qty
            total_drums_done += qty
            robot.drum_time += cycle_t

        robot.tool_time += tool_t
        robot.busy_time += dt
        robot.cycles += 1
        robot.last_tool = item_type

        heapq.heappush(evq, (t + dt, rid))

    makespan = max(r.next_free_t for r in robots)
    makespan = min(makespan, 1e8)

    per_robot_rows = []
    for r in robots:
        util = (r.busy_time / makespan) if makespan > 1e-12 else 0.0
        idle = max(0.0, 1.0 - util)
        box_frac = (r.box_time / makespan) if makespan > 1e-12 else 0.0
        drum_frac = (r.drum_time / makespan) if makespan > 1e-12 else 0.0
        tool_frac = (r.tool_time / makespan) if makespan > 1e-12 else 0.0
        per_robot_rows.append({
            "Robot": f"R{r.robot_id+1}",
            "Module": r.mtype.config_id,
            "Average utilisation": util,
            "Average idle time": idle,
            "Average re tooling time": tool_frac,
            "% time used for drums": drum_frac,
            "% time used for boxes": box_frac,
            "Cycles": r.cycles,
        })

    return makespan, total_drums_done, per_robot_rows


def simulate_layout_ratio(
    modules: Dict[str, ModuleType],
    layout_str: str,
    p: Tuple[float, float, float],
    n_orders: int,
) -> Tuple[float, float]:
    layout = parse_layout_str(layout_str)
    seed = abs(hash((layout_str, p, n_orders))) % 2_000_000_000
    seq = generate_order_sequence(p, int(n_orders), seed)

    makespan_s, drums_done, _ = simulate_sequence(layout, modules, seq)
    if makespan_s <= 1e-9:
        return 0.0, 0.0

    thr = (len(seq) / makespan_s) * 3600.0
    litres_hr = (drums_done * LITRES_PER_DRUM / makespan_s) * 3600.0
    return float(thr), float(litres_hr)


# -----------------------------
# Progress UI
# -----------------------------
class ProgressUI:
    def __init__(self):
        self.task = st.empty()
        self.meta = st.empty()
        self.bar = st.progress(0.0)
        self.t0 = time.perf_counter()
        self.last_ui = self.t0

    @staticmethod
    def _fmt(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "—"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds/60:.1f} min"
        return f"{seconds/3600:.2f} hr"

    def update(self, task_name: str, done: int, total: int):
        now = time.perf_counter()
        if (now - self.last_ui) < PROGRESS_UPDATE_SEC and done < total:
            return
        self.last_ui = now

        elapsed = now - self.t0
        rate = done / elapsed if elapsed > 1e-9 else 0.0
        remaining = (total - done) / rate if rate > 1e-9 else float("inf")
        frac = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))

        self.task.markdown(f"**Stage:** {task_name}")
        self.meta.markdown(
            f"**Runtime:** {self._fmt(elapsed)}  |  **Est remaining:** {self._fmt(remaining)}  |  **Tasks:** {done:,}/{total:,}"
        )
        self.bar.progress(frac)

    def finish(self):
        self.update("Done", 1, 1)


# -----------------------------
# Plot helpers
# -----------------------------
def make_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, selected_name: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(size=7),
            customdata=np.stack([df["layout_str"], df["robot_area"], df["cost"]], axis=1),
            hovertemplate=(
                "Layout: %{customdata[0]}<br>"
                "Cost: %{customdata[2]:,.0f}<br>"
                "Area: %{customdata[1]:.2f}<br>"
                f"{x_col}: %{{x}}<br>"
                f"{y_col}: %{{y}}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    sel = df[df["config_st_name"] == selected_name]
    if not sel.empty:
        fig.add_trace(
            go.Scatter(
                x=sel[x_col],
                y=sel[y_col],
                mode="markers",
                marker=dict(size=12, color="red"),
                customdata=np.stack([sel["layout_str"], sel["robot_area"], sel["cost"]], axis=1),
                hovertemplate=(
                    "SELECTED<br>"
                    "Layout: %{customdata[0]}<br>"
                    "Cost: %{customdata[2]:,.0f}<br>"
                    "Area: %{customdata[1]:.2f}<br>"
                    f"{x_col}: %{{x}}<br>"
                    f"{y_col}: %{{y}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_xaxes(title_text=x_col.replace("_", " ").title(), rangemode="tozero")
    fig.update_yaxes(title_text=y_col.replace("_", " ").title(), rangemode="tozero")

    fig.update_layout(
        title=title,
        height=PLOT_HEIGHT,
        dragmode=False,
        hovermode="closest",
        margin=dict(l=30, r=10, t=40, b=30),
    )
    return fig


def plot_square(df: pd.DataFrame, selected: str, specs: List[Tuple[str, str, str]]):
    r1c1, r1c2 = st.columns(2, gap="small")
    r2c1, r2c2 = st.columns(2, gap="small")

    with r1c1:
        st.plotly_chart(make_scatter(df, specs[0][0], specs[0][1], specs[0][2], selected),
                        width="stretch", config={"displayModeBar": False})
    with r1c2:
        st.plotly_chart(make_scatter(df, specs[1][0], specs[1][1], specs[1][2], selected),
                        width="stretch", config={"displayModeBar": False})
    with r2c1:
        st.plotly_chart(make_scatter(df, specs[2][0], specs[2][1], specs[2][2], selected),
                        width="stretch", config={"displayModeBar": False})
    with r2c2:
        st.plotly_chart(make_scatter(df, specs[3][0], specs[3][1], specs[3][2], selected),
                        width="stretch", config={"displayModeBar": False})


# -----------------------------
# Triangle heatmap (continuous)
# -----------------------------
def bary_to_xy(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = p1 * 0.0 + p2 * 1.0 + p3 * 0.5
    y = p1 * 0.0 + p2 * 0.0 + p3 * (math.sqrt(3) / 2.0)
    return x, y


def inside_triangle_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    h = math.sqrt(3) / 2.0
    m = math.sqrt(3)
    return (Y >= -1e-12) & (Y <= m * X + 1e-12) & (Y <= m * (1.0 - X) + 1e-12) & (Y <= h + 1e-12) & (X >= -1e-12) & (X <= 1.0 + 1e-12)


def idw_interpolate(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                    power: float = 2.0, eps: float = 1e-9) -> np.ndarray:
    gx = grid_x.reshape(-1, 1)
    gy = grid_y.reshape(-1, 1)
    dx = gx - xs.reshape(1, -1)
    dy = gy - ys.reshape(1, -1)
    d2 = dx * dx + dy * dy + eps
    w = 1.0 / (d2 ** (power / 2.0))
    z = (w @ zs.reshape(-1, 1)) / (np.sum(w, axis=1, keepdims=True))
    return z.reshape(grid_x.shape)


def make_triangle_heatmap(points_df: pd.DataFrame, title: str, zmin: float, zmax: float) -> go.Figure:
    p1 = points_df["p1"].to_numpy(dtype=float)
    p2 = points_df["p2"].to_numpy(dtype=float)
    p3 = points_df["p3"].to_numpy(dtype=float)
    z = points_df["throughput"].to_numpy(dtype=float)

    xs, ys = bary_to_xy(p1, p2, p3)

    grid_n = FINE_HEATMAP_GRID_N
    x_lin = np.linspace(0.0, 1.0, grid_n)
    y_lin = np.linspace(0.0, math.sqrt(3) / 2.0, grid_n)
    X, Y = np.meshgrid(x_lin, y_lin)

    mask = inside_triangle_mask(X, Y)
    Z = idw_interpolate(xs, ys, z, X, Y, power=2.0)
    Z_masked = np.where(mask, Z, np.nan)

    colorscale = [
        [0.0, "#67000d"],
        [0.5, "#fdae61"],
        [1.0, "#d9f0a3"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_lin,
            y=y_lin,
            z=Z_masked,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Throughput<br>(orders/hr)"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=5, color="rgba(0,0,0,0.25)"),
            customdata=np.stack([p1, p2, p3, z], axis=1),
            hovertemplate=(
                "Order 1: %{customdata[0]:.2f}<br>"
                "Order 2: %{customdata[1]:.2f}<br>"
                "Order 3: %{customdata[2]:.2f}<br>"
                "Throughput: %{customdata[3]:.2f} orders/hr<extra></extra>"
            ),
            showlegend=False,
        )
    )

    h = math.sqrt(3) / 2.0
    fig.add_trace(go.Scatter(x=[0.0, 1.0, 0.5, 0.0], y=[0.0, 0.0, h, 0.0], mode="lines",
                             line=dict(color="black"), showlegend=False))
    fig.add_annotation(x=0.0, y=0.0, text="Order 1", showarrow=False, xanchor="left", yanchor="top")
    fig.add_annotation(x=1.0, y=0.0, text="Order 2", showarrow=False, xanchor="right", yanchor="top")
    fig.add_annotation(x=0.5, y=h, text="Order 3", showarrow=False, xanchor="center", yanchor="bottom")

    fig.update_layout(
        title=title,
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        dragmode=False,
    )
    fig.update_xaxes(visible=False, range=[-0.02, 1.02])
    fig.update_yaxes(visible=False, range=[-0.02, h + 0.04], scaleanchor="x", scaleratio=1)

    return fig


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Packing Line Optimiser", layout="wide")
st.title("Robotic Packing Line Config Optimiser")

ratios_coarse = stepped_ratios(COARSE_STEP)

with st.sidebar:
    st.header("Inputs")

    with st.form("form", clear_on_submit=False):
        uploaded = st.file_uploader("Config Excel File", type=["xlsx"])
        use_defaults = st.checkbox("Use default Config File", value=(uploaded is None))

        max_area = st.number_input("Max Robot Area (m²)", min_value=0.1, value=16.0, step=1.0)

        n_orders = st.slider(
            "Orders per mixture point (random sequence)",
            min_value=20,
            max_value=400,   # lowered for cloud safety
            value=100,
            step=10,
            help="For each mixture (p1,p2,p3), we queue this many random orders and simulate tool switching/idle.",
        )

        run = st.form_submit_button("▶ Run")


# Load modules
try:
    if uploaded is not None and not use_defaults:
        modules = load_modules_from_excel_bytes(uploaded.getvalue())
    else:
        modules = default_modules()
except Exception as e:
    st.error(f"Failed to load module table. Falling back to defaults. Error: {e}")
    modules = default_modules()

df_mod = pd.DataFrame([{
    "Config ID": m.config_id,
    "Name": m.name,
    "Boxes/cycle (if box cycle)": m.box_per_cycle if m.can_box else 0,
    "Drums/cycle (if drum cycle)": m.drum_per_cycle if m.can_drum else 0,
    "Cycle time (s)": m.cycle_time_s,
    "Switch/Tool overhead (s)": m.toolchange_time_s,
    "Cost": m.cost,
    "Robot area (m²)": m.area,
    "Mode": m.mode,
} for m in modules.values()]).sort_values("Config ID")

with st.expander("Module types", expanded=False):
    st.dataframe(df_mod, width="stretch")

# State
if "results" not in st.session_state:
    st.session_state.results = None
if "global_thr_minmax" not in st.session_state:
    st.session_state.global_thr_minmax = None

# Run
if run:
    try:
        prog = ProgressUI()

        prog.update("Enumerating layouts under max area (capped)", 0, 1)
        layouts = enumerate_layouts(modules, float(max_area))
        feasible = [lay for lay in layouts if layout_can_do_mixed(lay, modules)]

        if not feasible:
            st.session_state.results = pd.DataFrame()
            st.session_state.global_thr_minmax = (0.0, 1.0)
            prog.finish()
        else:
            # Convert to strings
            feasible_strs = [layout_to_str(lay) for lay in feasible]

            # If too many layouts, sample to keep Cloud alive.
            # Heuristic: prefer lower area first (usually more realistic & less robots).
            if len(feasible_strs) > MAX_EVALUATED_LAYOUTS:
                # stable ordering by (area, cost) then take first chunk (deterministic)
                scored = []
                for ls in feasible_strs:
                    lay = parse_layout_str(ls)
                    cost, area = compute_layout_cost_area(lay, modules)
                    scored.append((area, cost, ls))
                scored.sort()
                feasible_strs = [t[2] for t in scored[:MAX_EVALUATED_LAYOUTS]]

                st.warning(
                    f"Too many feasible layouts for Cloud limits. "
                    f"Evaluating the best {MAX_EVALUATED_LAYOUTS} layouts by (area, cost)."
                )

            total_tasks = len(feasible_strs) * len(ratios_coarse)
            done = 0
            prog.update("Simulating layouts × mixture ratios", done, total_tasks)

            rows = []
            global_min_thr = float("inf")
            global_max_thr = float("-inf")

            for ls in feasible_strs:
                lay = parse_layout_str(ls)
                cost, area = compute_layout_cost_area(lay, modules)

                thr_list = []
                lph_list = []
                for p in ratios_coarse:
                    thr, lph = simulate_layout_ratio(modules, ls, p, int(n_orders))
                    thr_list.append(thr)
                    lph_list.append(lph)

                    global_min_thr = min(global_min_thr, thr)
                    global_max_thr = max(global_max_thr, thr)

                    done += 1
                    prog.update("Simulating layouts × mixture ratios", done, total_tasks)

                thr_arr = np.array(thr_list, dtype=float)
                lph_arr = np.array(lph_list, dtype=float)

                rows.append({
                    "layout_str": ls,
                    "config_st_name": ls,
                    "cost": float(cost),
                    "robot_area": float(area),
                    "max_throughput": float(np.max(thr_arr)),
                    "min_throughput": float(np.min(thr_arr)),
                    "median_throughput": float(np.median(thr_arr)),
                    "median_litres_per_hr": float(np.median(lph_arr)),
                })

            df = pd.DataFrame(rows).sort_values("median_throughput", ascending=False).reset_index(drop=True)

            if not math.isfinite(global_min_thr) or not math.isfinite(global_max_thr):
                global_min_thr, global_max_thr = 0.0, 1.0
            if abs(global_max_thr - global_min_thr) < 1e-12:
                global_max_thr = global_min_thr + 1.0

            st.session_state.results = df
            st.session_state.global_thr_minmax = (float(global_min_thr), float(global_max_thr))
            prog.finish()

    except Exception as e:
        st.session_state.results = None
        st.session_state.global_thr_minmax = None
        st.error("Run failed with an exception. Details below:")
        st.exception(e)

# Show results
df = st.session_state.results
if df is None:
    st.info("Set inputs and press **Run**.")
    st.stop()

if df.empty:
    st.warning("No feasible mixed-capable layouts found under the given max area.")
    st.stop()

st.subheader("Results Ranking table")
st.dataframe(df.head(200), width="stretch")

st.subheader("Select Configuration (red)")
selected = st.selectbox("Configuration", df["config_st_name"].tolist(), index=0)

st.markdown("### Throughput vs Cost")
plot_square(
    df, selected,
    specs=[
        ("cost", "max_throughput", "Max Throughput vs Cost"),
        ("cost", "min_throughput", "Min Throughput vs Cost"),
        ("cost", "median_throughput", "Median Throughput vs Cost"),
        ("cost", "median_litres_per_hr", "Median Litres per hour vs Cost"),
    ],
)
st.markdown("**Legend**: red dot = selected configuration.")

# Heatmap for selected config (fine grid) with GLOBAL scale
st.markdown("### Order Mixture Ratio Heatmap")
global_min_thr, global_max_thr = st.session_state.global_thr_minmax or (0.0, 1.0)

ratios_fine = stepped_ratios(FINE_STEP_FOR_SELECTED)

sel_points = []
for p in ratios_fine:
    thr, _lph = simulate_layout_ratio(modules, selected, p, int(n_orders))
    sel_points.append({"p1": p[0], "p2": p[1], "p3": p[2], "throughput": thr})

sel_df = pd.DataFrame(sel_points)
st.plotly_chart(
    make_triangle_heatmap(
        sel_df,
        title="Order-ratio heatmap with global colour scale",
        zmin=global_min_thr,
        zmax=global_max_thr,
    ),
    width="stretch",
    config={"displayModeBar": False},
)

# Robot KPI table
st.subheader("Robot KPI's")
sel_layout = parse_layout_str(selected)
robots = build_robot_instances(sel_layout, modules)
if not robots:
    st.info("No robots found (unexpected).")
    st.stop()

acc = {f"R{i+1}": {
    "Robot": f"R{i+1}",
    "Module": robots[i].mtype.config_id,
    "Average utilisation": 0.0,
    "Average idle time": 0.0,
    "Average re tooling time": 0.0,
    "% time used for drums": 0.0,
    "% time used for boxes": 0.0,
    "Cycles": 0.0,
} for i in range(len(robots))}

for p in ratios_coarse:
    seed = abs(hash((selected, p, int(n_orders), "robot_kpis"))) % 2_000_000_000
    seq = generate_order_sequence(p, int(n_orders), seed)
    makespan_s, _drums_done, per_rows = simulate_sequence(sel_layout, modules, seq)
    if makespan_s <= 1e-9:
        continue
    for row in per_rows:
        rname = row["Robot"]
        if rname in acc:
            for k in ["Average utilisation", "Average idle time", "Average re tooling time",
                      "% time used for drums", "% time used for boxes"]:
                acc[rname][k] += float(row[k])
            acc[rname]["Cycles"] += float(row["Cycles"])

nR = len(ratios_coarse)
out_rows = []
for _, row in acc.items():
    row2 = dict(row)
    for k in ["Average utilisation", "Average idle time", "Average re tooling time",
              "% time used for drums", "% time used for boxes"]:
        row2[k] = row2[k] / max(1, nR)
    row2["Cycles"] = row2["Cycles"] / max(1, nR)
    out_rows.append(row2)

kpi_inst = pd.DataFrame(out_rows).sort_values(["Module", "Robot"]).reset_index(drop=True)

show = kpi_inst.copy()
pct_cols = [
    "Average utilisation",
    "Average idle time",
    "Average re tooling time",
    "% time used for drums",
    "% time used for boxes",
]
for c in pct_cols:
    show[c] = (show[c] * 100.0).round(2)
show["Cycles"] = show["Cycles"].round(1)

st.dataframe(show, width="stretch")
