"""
구역별 데이터 분리: 도면 구역(floors.json 의 zones)마다 Wi-Fi·BT 스캔을 나눈다

사용법:
  python zone_split.py              # data/WifiBleCollector 에서 핑이 있고, 도면에 구역이 있는 세션 전부
  python zone_split.py 세션 [세션 ...]

구역 라벨 두 가지 (스캔마다):
  zone_gt   : GT 궤적(핑 + IMU 보행 추정, gt_reconstruct.py) 기준.
              스캔이 걸린 시간 동안의 위치가 모두 한 구역 안에 있고, 경계에서 MARGIN_M 이상 안쪽일 때만 붙인다.
  zone_ping : 사람이 찍은 핑만 기준 (IMU 안 씀, 크라우드소싱 방식).
              핑 전후 STAND_S 초 안의 스캔 → 그 핑의 구역. 앞뒤 핑이 같은 구역이면 그 사이 스캔도 그 구역.
              첫 핑 전 / 마지막 핑 후는 그 핑 구역 (출발·도착점에 서 있는 것으로 본다).
  두 라벨 공통: 첫 핑 전 / 마지막 핑 후에 걸음이 검출되면 그 구간은 라벨을 붙이지 않는다.
  라벨을 못 붙이면 빈칸이고, 이유는 gt_note / ping_note 열에 적는다.

출력 (output/<세션>/):
  zone_wifi.csv, zone_ble.csv : 원래 행 + x_m, y_m, zone_gt, zone_ping
  zone_summary.csv            : 구역별 스캔 수 (GT / 핑 / 둘이 일치)
  zone_split.png              : 도면 위 구역·궤적, 구역별 스캔 수, GT 대 핑 혼동 행렬
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gt_reconstruct import detect_steps, load, reconstruct

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
# 세션 폴더는 data/WifiBleCollector/ 또는 data/ 바로 아래에 둔다
DEFAULT_DATA = BASE / "data" / "WifiBleCollector" if (BASE / "data" / "WifiBleCollector").is_dir() else BASE / "data"
DEFAULT_OUT = BASE / "output"
FLOORS_JSON = Path(r"D:\AndroidProjects\app\src\main\assets\floorplans\floors.json")

WIFI_SPAN_S = 2.0   # Wi-Fi 스캔 1회가 걸리는 시간으로 본다 (결과 시각 이전 2초 동안의 위치를 모두 본다)
BLE_SPAN_S = 1.0    # BT 는 1초 단위로 묶어 저장된다
MARGIN_M = 0.3      # GT 라벨: 구역 경계에서 이만큼 안쪽이어야 한다
STAND_S = 2.0       # 핑 라벨: 핑 전후 이 시간은 그 자리에 서 있던 것으로 본다
SAMPLES = 5         # 스캔 시간 동안 위치를 몇 점에서 볼지

ROOM = "#2F5BD3"
CORRIDOR = "#D9480F"
GRID = "#E8E8E8"
NONE = "(없음)"


# ── 구역 ──
def load_zones(floor_name: str, m_per_px: float) -> list:
    """[{name, type, poly (N×2, m)}]. 해당 층에 구역이 없으면 []"""
    cfg = json.loads(FLOORS_JSON.read_text(encoding="utf-8"))
    floor = next((f for f in cfg["floors"] if f["name"] == floor_name), None)
    if floor is None:
        return []
    return [{"name": z["name"], "type": z.get("type", ""), "poly": np.array(z["polygon"], float) * m_per_px}
            for z in floor.get("zones", [])]


def inside(x, y, poly) -> np.ndarray:
    """점이 다각형 안에 있는지 (ray casting)"""
    res = np.zeros(np.shape(x), bool)
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        cross = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        res ^= cross & (x < xi)
    return res


def edge_distance(x, y, poly) -> np.ndarray:
    """다각형 변까지 최단 거리"""
    d = np.full(np.shape(x), np.inf)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1])))
    return d


def centroid(poly) -> np.ndarray:
    """다각형 무게중심 (ㄱ자 구역도 안쪽에 라벨이 오도록)"""
    x, y = poly[:, 0], poly[:, 1]
    c = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = c.sum() / 2
    return np.array([((x + np.roll(x, -1)) * c).sum(), ((y + np.roll(y, -1)) * c).sum()]) / (6 * a)


def zone_of(x, y, zones):
    """(구역 이름 배열, 경계까지 거리 배열). 구역 밖은 '' / 0"""
    name = np.full(np.shape(x), "", dtype=object)
    dist = np.zeros(np.shape(x))
    for z in zones:
        m = inside(x, y, z["poly"])
        name[m] = z["name"]
        dist[m] = edge_distance(x[m], y[m], z["poly"])
    return name, dist


# ── 라벨 ──
def position(track: pd.DataFrame, t):
    tt = track.timestamp.to_numpy()
    return np.interp(t, tt, track.x_m), np.interp(t, tt, track.y_m)


def label_gt(scan_t: np.ndarray, span_s: float, track, zones) -> pd.DataFrame:
    # 스캔 결과 시각 이전 span 동안 SAMPLES 개 시점의 위치
    ts = scan_t[:, None] - np.linspace(span_s * 1000, 0, SAMPLES)[None, :]
    x, y = position(track, ts)
    names, dist = zone_of(x, y, zones)
    first = names[:, 0]
    same = (names == first[:, None]).all(axis=1)
    deep = dist.min(axis=1) >= MARGIN_M
    zone = np.where(same & (first != "") & deep, first, "")
    note = np.select(
        [zone != "", ~same, first == ""],
        ["", "스캔 중 구역 이동", "구역 밖"],
        default="경계 가까움",
    )
    xe, ye = position(track, scan_t)
    return pd.DataFrame({"timestamp": scan_t, "x_m": xe.round(3), "y_m": ye.round(3), "zone_gt": zone, "gt_note": note})


def label_ping(scan_t: np.ndarray, span_s: float, gt: pd.DataFrame, zones) -> pd.DataFrame:
    pt = gt.timestamp.to_numpy()
    pz, _ = zone_of(gt.x_m.to_numpy(), gt.y_m.to_numpy(), zones)
    zone, note = [], []
    for t in scan_t:
        lo, hi = t - span_s * 1000, t
        # 스캔 시간과 핑 전후 STAND_S 가 겹치면 그 핑 자리에 서 있던 것
        near = np.where((pt + STAND_S * 1000 >= lo) & (pt - STAND_S * 1000 <= hi))[0]
        if len(near):
            k = near[np.argmin(np.abs(pt[near] - t))]
            zone.append(pz[k])
            note.append("" if pz[k] else "핑이 구역 밖")
            continue
        prev = np.where(pt < lo)[0]
        nxt = np.where(pt > hi)[0]
        if len(prev) and len(nxt):
            a, b = pz[prev[-1]], pz[nxt[0]]
            if a and a == b:
                zone.append(a)
                note.append("")
            else:
                zone.append("")
                note.append("앞뒤 핑 구역이 다름" if a or b else "핑이 구역 밖")
        else:
            # 첫 핑 전 / 마지막 핑 후는 그 핑 자리에 서 있던 것으로 본다 (GT 궤적과 같은 가정)
            k = nxt[0] if len(nxt) else prev[-1]
            zone.append(pz[k])
            note.append("" if pz[k] else "핑이 구역 밖")
    return pd.DataFrame({"timestamp": scan_t, "zone_ping": zone, "ping_note": note})


def drop_unanchored(lab: pd.DataFrame, span_s: float, gt: pd.DataFrame, step_t: np.ndarray) -> pd.DataFrame:
    """첫 핑 전 / 마지막 핑 후에 걸음이 있으면 '그 핑 자리에 서 있었다'는 가정이 틀리므로 라벨을 지운다"""
    first, last = gt.timestamp.min(), gt.timestamp.max()
    t = lab.timestamp.to_numpy()
    before = np.array([((step_t > x - span_s * 1000) & (step_t < first)).any() for x in t]) & (t < first)
    after = np.array([((step_t > last) & (step_t <= x)).any() for x in t]) & (t > last)
    for col, note in [("zone_gt", "gt_note"), ("zone_ping", "ping_note")]:
        lab.loc[before, col] = ""
        lab.loc[before, note] = "첫 핑 전 이동"
        lab.loc[after, col] = ""
        lab.loc[after, note] = "마지막 핑 후 이동"
    return lab


def label_scans(df: pd.DataFrame, span_s: float, track, gt, zones, step_t) -> pd.DataFrame:
    scan_t = np.sort(df.timestamp.unique()).astype(float)
    lab = label_gt(scan_t, span_s, track, zones).merge(label_ping(scan_t, span_s, gt, zones), on="timestamp")
    lab = drop_unanchored(lab, span_s, gt, step_t)
    lab["timestamp"] = lab.timestamp.astype(df.timestamp.dtype)
    return df.merge(lab, on="timestamp", how="left")


# ── 요약 / 그림 ──
def summarize(wifi_lab, ble_lab, zones) -> pd.DataFrame:
    rows = []
    for z in zones:
        r = {"구역": z["name"], "종류": {"room": "방", "corridor": "복도"}.get(z["type"], z["type"])}
        for name, lab in [("Wi-Fi", wifi_lab), ("BT", ble_lab)]:
            if lab is None or not len(lab):
                continue
            s = lab.drop_duplicates("timestamp")
            r[f"{name} 스캔 (GT)"] = int((s.zone_gt == z["name"]).sum())
            r[f"{name} 스캔 (핑)"] = int((s.zone_ping == z["name"]).sum())
            r[f"{name} 일치"] = int(((s.zone_gt == z["name"]) & (s.zone_ping == z["name"])).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def agreement(lab: pd.DataFrame) -> dict:
    s = lab.drop_duplicates("timestamp")
    both = (s.zone_gt != "") & (s.zone_ping != "")
    return {
        "스캔": len(s),
        "GT 라벨": int((s.zone_gt != "").sum()),
        "핑 라벨": int((s.zone_ping != "").sum()),
        "둘 다": int(both.sum()),
        "일치율 (%)": round(100 * (s.zone_gt[both] == s.zone_ping[both]).mean(), 1) if both.any() else np.nan,
    }


def plot(name, d, meta, gt, track, zones, wifi_lab, ble_lab, summary, out: Path):
    k = float(meta["m_per_px"])
    img = plt.imread(d / "floorplan.jpg")
    allp = np.vstack([z["poly"] for z in zones])
    pad = 3
    x0, y0 = allp.min(0) - pad
    x1, y1 = allp.max(0) + pad

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], hspace=0.35, wspace=0.3, top=0.9)

    # (a) 도면 + 구역 + GT 궤적
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, extent=[0, img.shape[1] * k, img.shape[0] * k, 0])
    for z in zones:
        c = ROOM if z["type"] == "room" else CORRIDOR
        ax.add_patch(plt.Polygon(z["poly"], closed=True, fc=c, ec=c, alpha=0.18, lw=1.5))
        ax.add_patch(plt.Polygon(z["poly"], closed=True, fill=False, ec=c, lw=1.5))
        cx, cy = centroid(z["poly"])
        ax.text(cx, cy, z["name"].replace("area ", ""), ha="center", va="center", fontsize=10, fontweight="bold", color=c)
    ax.plot(track.x_m, track.y_m, color="#333", lw=1, alpha=0.6, label="GT 궤적")
    lab = ble_lab if ble_lab is not None and len(ble_lab) else wifi_lab
    s = lab.drop_duplicates("timestamp")
    ok = s.zone_gt != ""
    ax.scatter(s.x_m[~ok], s.y_m[~ok], s=10, color="#999", label="GT 라벨 없음 (경계·이동·구역 밖)", zorder=3)
    ax.scatter(s.x_m[ok], s.y_m[ok], s=10, color="#1B7F3B", label="GT 라벨 있음", zorder=3)
    ax.scatter(gt.x_m, gt.y_m, s=60, color="#D99A00", ec="white", zorder=4, label="핑")
    for _, p in gt.iterrows():
        ax.annotate(str(int(p.seq)), (p.x_m, p.y_m), ha="center", va="center", fontsize=7, color="white", zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"(a) 구역과 GT 궤적 ({'BT' if lab is ble_lab else 'Wi-Fi'} 스캔 위치)", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)

    # (b)(c) 구역별 스캔 수
    names = [z["name"].replace("area ", "") for z in zones]
    for col, src in enumerate(["Wi-Fi", "BT"]):
        ax = fig.add_subplot(gs[0, 1 + col])
        ax.set_title(f"(b{col + 1}) 구역별 {src} 스캔 수", loc="left", fontsize=11)
        if f"{src} 스캔 (GT)" in summary:
            x = np.arange(len(zones))
            ax.bar(x - 0.2, summary[f"{src} 스캔 (GT)"], 0.4, color="#1B7F3B", label="GT 기준")
            ax.bar(x + 0.2, summary[f"{src} 스캔 (핑)"], 0.4, color="#D99A00", label="핑 기준")
            ax.set_xticks(x, names)
            ax.set_xlabel("구역 (area 번호)")
            ax.legend(fontsize=8.5)
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)

    # (d) 혼동 행렬 (BT 가 있으면 BT, 없으면 Wi-Fi)
    ax = fig.add_subplot(gs[1, 1:])
    s = lab.drop_duplicates("timestamp")
    order = [z["name"] for z in zones] + [""]
    cm = pd.crosstab(s.zone_gt, s.zone_ping).reindex(index=order, columns=order, fill_value=0)
    lbl = [n.replace("area ", "") if n else NONE for n in order]
    ax.imshow(np.log1p(cm.values), cmap="Greens")
    for i in range(len(order)):
        for j in range(len(order)):
            v = cm.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if v > cm.values.max() * 0.4 else "#222")
    ax.set_xticks(range(len(order)), lbl)
    ax.set_yticks(range(len(order)), lbl)
    ax.set_xlabel("핑 기준 구역")
    ax.set_ylabel("GT 기준 구역")
    a = agreement(lab)
    ax.set_title(f"(c) GT 대 핑 구역 ({'BT' if lab is ble_lab else 'Wi-Fi'} 스캔 수) · 둘 다 라벨 있는 스캔 중 일치 {a['일치율 (%)']}%",
                 loc="left", fontsize=11)

    fig.suptitle(f"구역별 데이터 분리 — {name}", fontsize=14, fontweight="bold", x=0.08, ha="left", y=0.965)
    fig.text(0.08, 0.93, f"GT 라벨: 스캔 시간(Wi-Fi {WIFI_SPAN_S:.0f}초, BT {BLE_SPAN_S:.0f}초) 동안 한 구역 안 + 경계에서 {MARGIN_M} m 이상 안쪽 · "
             f"핑 라벨: 핑 전후 {STAND_S:.0f}초, 또는 앞뒤 핑이 같은 구역", fontsize=10, color="#555")
    fig.savefig(out / "zone_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_session(d: Path, out_root: Path):
    meta, gt, imu, wifi = load(d)
    zones = load_zones(str(meta.get("floorplan_name", "")), float(meta["m_per_px"]))
    if not zones:
        print(f"[건너뜀] {d.name}: 도면 '{meta.get('floorplan_name')}' 에 구역이 없습니다 ({FLOORS_JSON})")
        return
    if len(gt) < 1:
        print(f"[건너뜀] {d.name}: 핑이 없습니다")
        return
    track, _, _, _ = reconstruct(gt, imu) if len(gt) >= 2 else (gt[["timestamp", "x_m", "y_m"]], None, None, None)
    ble = pd.read_csv(d / "ble.csv", dtype={"peer": str}) if (d / "ble.csv").exists() else None

    step_t, _ = detect_steps(imu)
    wifi_lab = label_scans(wifi, WIFI_SPAN_S, track, gt, zones, step_t)
    ble_lab = label_scans(ble, BLE_SPAN_S, track, gt, zones, step_t) if ble is not None and len(ble) else None
    summary = summarize(wifi_lab, ble_lab, zones)

    out = out_root / d.name
    out.mkdir(parents=True, exist_ok=True)
    wifi_lab.to_csv(out / "zone_wifi.csv", index=False, encoding="utf-8-sig")
    if ble_lab is not None:
        ble_lab.to_csv(out / "zone_ble.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "zone_summary.csv", index=False, encoding="utf-8-sig")
    plot(d.name, d, meta, gt, track, zones, wifi_lab, ble_lab, summary, out)

    print(f"\n{d.name}  (핑 {len(gt)}개, 구역 {len(zones)}개)")
    print(summary.to_string(index=False))
    for name, lab in [("Wi-Fi", wifi_lab), ("BT", ble_lab)]:
        if lab is not None:
            a = agreement(lab)
            notes = lab.drop_duplicates("timestamp").gt_note.value_counts().drop("", errors="ignore").to_dict()
            print(f"  {name}: {a} · GT 라벨 못 붙인 이유 {notes}")
    print(f"  → {out}")


def has_pings(d: Path) -> bool:
    try:
        return len(pd.read_csv(d / "gt.csv")) > 0
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False


if __name__ == "__main__":
    args = sys.argv[1:]
    dirs = [Path(a) if Path(a).is_dir() else DEFAULT_DATA / a for a in args] if args \
        else [d for d in sorted(DEFAULT_DATA.iterdir()) if has_pings(d)]
    if not dirs:
        print("핑이 있는 세션이 없습니다")
    for d in dirs:
        run_session(d, DEFAULT_OUT)
