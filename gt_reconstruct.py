"""
위치 정답(GT) 궤적 복원: 도면 핑(gt.csv) + IMU 보행 추정(PDR) → 앵커 보정 → Wi-Fi 스캔에 좌표 부여

입력: data/WifiBleCollector/<세션>_gt/  (gt.csv, imu.csv, wifi.csv, meta.csv, floorplan.jpg)
처리:
    1) 걸음 검출: 가속도 크기(이동평균) 피크, 최소 간격 0.3 s
    2) 보폭: Weinberg L = K (a_max - a_min)^(1/4)
    3) 방향: game rotation vector 의 yaw (자기장 안 씀)
    4) 핑 사이 구간마다 PDR 적분 → 닮음 변환(회전 θ, 배율 s)으로 끝점을 다음 핑에 맞춤
       첫 핑 전 / 마지막 핑 후는 그 핑 위치에 서 있는 것으로 본다
    5) Wi-Fi 스캔 시각의 위치를 보정 궤적에서 선형 보간
출력 (output/<세션>/):
    gt_track.csv      : 걸음마다 timestamp, x_m, y_m
    wifi_labeled.csv  : timestamp, bssid, rssi, floor, x_m, y_m  (핑거프린트 DB 원재료)
    gt_segments.csv   : 구간별 PDR 거리, 핑 거리, θ, s, 품질
    gt.png            : 도면 위 GT 궤적 + 시간축
좌표: 도면 이미지 왼쪽 위가 원점, x 오른쪽 / y 아래쪽 (m)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
# 세션 폴더는 data/WifiBleCollector/ 또는 data/ 바로 아래에 둔다
DEFAULT_DATA = BASE / "data" / "WifiBleCollector" if (BASE / "data" / "WifiBleCollector").is_dir() else BASE / "data"
DEFAULT_OUT = BASE / "output"

STEP_K = 0.47          # Weinberg 상수 (사람마다 다름, 배율 s 로 보정됨)
STEP_THRESH = 10.6     # 걸음 피크 최소 가속도 크기 (m/s²). 10.8 이면 천천히 걸을 때 걸음을 놓친다
STEP_MIN_GAP = 0.3     # 걸음 사이 최소 간격 (s)
MAX_THETA_DEG = 15     # 품질 검사
SCALE_RANGE = (0.8, 1.2)


def load(d: Path):
    meta = pd.read_csv(d / "meta.csv").set_index("key")["value"].to_dict()
    gt = pd.read_csv(d / "gt.csv").sort_values("timestamp")
    imu = pd.read_csv(d / "imu.csv")
    wifi = pd.read_csv(d / "wifi.csv", sep=r"\s*,\s*", engine="python")
    wifi.columns = wifi.columns.str.strip()
    return meta, gt, imu, wifi


def detect_steps(imu: pd.DataFrame):
    """(걸음 시각 ms 배열, 걸음별 보폭 m 배열)"""
    acc = imu[imu.sensor == "acc"].sort_values("timestamp")
    t = acc.timestamp.to_numpy()
    mag = np.sqrt(acc.x**2 + acc.y**2 + acc.z**2).to_numpy()
    f = np.convolve(mag, np.ones(7) / 7, mode="same")

    peaks = []
    for i in range(1, len(f) - 1):
        if f[i] > STEP_THRESH and f[i] > f[i - 1] and f[i] >= f[i + 1]:
            if peaks and (t[i] - t[peaks[-1]]) / 1000 < STEP_MIN_GAP:
                if f[i] > f[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    peaks = np.array(peaks, dtype=int)

    # 보폭: 이전 걸음 ~ 이번 걸음 사이 가속도 최대-최소
    lengths = []
    for j, p in enumerate(peaks):
        lo = peaks[j - 1] if j > 0 else max(0, p - 30)
        seg = f[lo:p + 1]
        lengths.append(STEP_K * max(seg.max() - seg.min(), 0) ** 0.25)
    return t[peaks], np.array(lengths)


def heading_at(imu: pd.DataFrame, times_ms: np.ndarray) -> np.ndarray:
    """game rotation vector 의 yaw (rad). 원점 방향은 임의 → 구간 회전 θ 로 맞춰진다."""
    rv = imu[imu.sensor == "game_rv"].sort_values("timestamp")
    x, y, z, w = (rv[c].to_numpy() for c in ("x", "y", "z", "w"))
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))
    return np.interp(times_ms, rv.timestamp.to_numpy(), yaw)


def rot(v, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


def reconstruct(gt, imu):
    """보정 궤적(DataFrame), 보정 전 PDR(DataFrame), 구간 표(DataFrame)"""
    step_t, step_len = detect_steps(imu)
    yaw = heading_at(imu, step_t)
    # 이미지 좌표는 y 가 아래쪽이라 y 성분 부호를 뒤집어야 좌/우 회전 방향이 유지된다
    d = np.stack([step_len * np.cos(yaw), -step_len * np.sin(yaw)], axis=1)

    pt = gt.timestamp.to_numpy()
    anchors = gt[["x_m", "y_m"]].to_numpy()
    track = [(pt[0], *anchors[0])]
    raw = [(pt[0], *anchors[0])]
    rows = []
    raw_pos = anchors[0].copy()
    prev_th = None
    for i in range(len(pt) - 1):
        a, b = anchors[i], anchors[i + 1]
        m = (step_t > pt[i]) & (step_t <= pt[i + 1])
        ts, ds = step_t[m], d[m]
        p = np.cumsum(ds, axis=0) if len(ds) else np.zeros((0, 2))
        pdr_end = p[-1] if len(p) else np.zeros(2)
        ab = b - a
        pdr_len, gt_len = float(np.hypot(*pdr_end)), float(np.hypot(*ab))

        if pdr_len > 0.5 and gt_len > 0.5:
            th = np.arctan2(ab[1], ab[0]) - np.arctan2(pdr_end[1], pdr_end[0])
            th = (th + np.pi) % (2 * np.pi) - np.pi
            s = gt_len / pdr_len
            q = a + s * rot(p, th) if len(p) else np.zeros((0, 2))
            mode = "닮음 변환"
        else:
            # 제자리 복귀 또는 거의 안 움직인 구간: 오차를 시간 비율로 나눠 더함
            th, s = 0.0, 1.0
            frac = (ts - pt[i]) / (pt[i + 1] - pt[i]) if len(ts) else np.zeros(0)
            q = a + p + frac[:, None] * (ab - pdr_end) if len(p) else np.zeros((0, 2))
            mode = "선형 분배"

        # yaw 의 기준 방향은 임의라 첫 구간 θ 는 '초기 방향 맞춤'일 뿐이다.
        # 방향 오차는 직전 구간 대비 θ 변화(dθ)로 본다.
        dth = None if prev_th is None or mode != "닮음 변환" else (th - prev_th + np.pi) % (2 * np.pi) - np.pi
        if mode == "닮음 변환":
            prev_th = th
        ok = (dth is None or abs(np.degrees(dth)) <= MAX_THETA_DEG) and SCALE_RANGE[0] <= s <= SCALE_RANGE[1]
        rows.append({
            "구간": f"{i + 1}→{i + 2}",
            "시간_s": round((pt[i + 1] - pt[i]) / 1000, 1),
            "걸음수": int(m.sum()),
            "PDR거리_m": round(pdr_len, 2),
            "핑거리_m": round(gt_len, 2),
            "끝점오차_m": round(float(np.hypot(*(a + pdr_end - b))), 2),
            "θ_deg": round(float(np.degrees(th)), 1),
            "dθ_deg": "-" if dth is None else round(float(np.degrees(dth)), 1),
            "s": round(float(s), 3),
            "방식": mode,
            "품질": "OK" if ok else "확인 필요",
        })
        track += [(t_, *xy) for t_, xy in zip(ts, q)]
        track.append((pt[i + 1], *b))
        raw += [(t_, *(raw_pos + xy)) for t_, xy in zip(ts, p)]
        raw_pos = raw_pos + pdr_end

    cols = ["timestamp", "x_m", "y_m"]
    return pd.DataFrame(track, columns=cols), pd.DataFrame(raw, columns=cols), pd.DataFrame(rows), step_t


def label_wifi(wifi, track, gt):
    """스캔 시각 위치 보간. 첫 핑 전 / 마지막 핑 후 스캔은 그 핑 위치 (서 있는 구간)"""
    t = track.timestamp.to_numpy()
    out = wifi.copy()
    out["floor"] = gt.floor.iloc[0]
    out["x_m"] = np.interp(out.timestamp, t, track.x_m).round(3)
    out["y_m"] = np.interp(out.timestamp, t, track.y_m).round(3)
    return out[["timestamp", "bssid", "rssi", "floor", "x_m", "y_m"]]


def plot(name, d, meta, gt, imu, wifi_lab, track, raw, seg, step_t, out: Path):
    k = float(meta["m_per_px"])
    img = plt.imread(d / "floorplan.jpg")
    h, w = img.shape[:2]
    t0 = float(meta["start_wall_ms"])
    scans = wifi_lab.drop_duplicates("timestamp")

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 0.8, 0.75], width_ratios=[1, 1], hspace=0.3, wspace=0.15)

    def draw_map(ax, zoom):
        ax.imshow(img, extent=[0, w * k, h * k, 0], alpha=0.85)
        ax.plot(raw.x_m, raw.y_m, ":", color="#C44E52", lw=1.6, label="IMU만 (보정 전)")
        ax.plot(track.x_m, track.y_m, "-", color="#0F7B5F", lw=2.4 if zoom else 1.6, label="GT (앵커 보정)")
        ax.scatter(scans.x_m, scans.y_m, s=34 if zoom else 10, marker="s", color="#2F5BD3",
                   edgecolor="white", lw=0.5, zorder=4, label=f"Wi-Fi 스캔 위치 ({len(scans)}회)")
        ax.scatter(gt.x_m, gt.y_m, s=230 if zoom else 70, color="#D99A00", edgecolor="white", lw=1.2, zorder=5,
                   label="핑 (번호 = 찍은 순서)")
        if zoom:
            for _, r in gt.iterrows():
                ax.text(r.x_m, r.y_m, str(int(r.seq)), ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold", zorder=6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    ax = fig.add_subplot(gs[0, 0])
    draw_map(ax, zoom=False)
    xs = np.r_[track.x_m, raw.x_m]
    ys = np.r_[track.y_m, raw.y_m]
    pad = max(np.ptp(xs), np.ptp(ys), 5) * 0.12
    box = (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)
    ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False,
                               ec="#333", ls="--", lw=1))
    ax.set_title(f"(a) 전체 도면: {meta.get('floorplan_name', '')}\n축척 1 px = {k:.4f} m · 점선 상자 = (b) 확대 영역",
                 loc="left", fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    draw_map(ax, zoom=True)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[3], box[2])
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    ax.set_title("(b) 확대: GT 궤적과 Wi-Fi 스캔 위치", loc="left", fontsize=11)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("(c) 구간별 보정 결과", loc="left", fontsize=11)
    cols = ["구간", "시간_s", "걸음수", "PDR거리_m", "핑거리_m", "dθ_deg", "s", "품질"]
    tb = ax.table(cellText=seg[cols].values, colLabels=["구간", "시간(s)", "걸음", "PDR(m)", "핑거리(m)", "dθ(°)", "s", "품질"],
                  bbox=[0, 0, 1, 1], cellLoc="center", colWidths=[0.1, 0.11, 0.09, 0.12, 0.14, 0.1, 0.1, 0.16])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#C9D2CF")
        if r == 0:
            cell.set_facecolor("#E8EEEC")
        elif c == len(cols) - 1 and cell.get_text().get_text() != "OK":
            cell.get_text().set_color("#B3261E")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    n_ok = int((seg.품질 == "OK").sum())
    notes = [
        f"세션 길이 {float(meta['duration_s']):.1f} s · 핑 {len(gt)}개 · 걸음 {len(step_t)}회 검출",
        f"Wi-Fi 신규 스캔 {len(scans)}회 → 좌표 부여 {len(wifi_lab)}행 (BSSID {wifi_lab.bssid.nunique()}개)",
        f"품질 OK 구간 {n_ok} / {len(seg)}",
        "",
        "dθ: 직전 구간 대비 방향 보정 변화 (첫 구간은 기준이라 -)",
        "s: 핑 거리 ÷ PDR 거리 (1이면 보폭 추정이 정확)",
        f"품질 기준: |dθ| ≤ {MAX_THETA_DEG}°, {SCALE_RANGE[0]} ≤ s ≤ {SCALE_RANGE[1]}",
    ]
    if n_ok < len(seg):
        notes += ["", "확인 필요 구간: 핑 위치, 걸음 검출(천천히 걷기·제자리 움직임)을 점검"]
    ax.text(0.03, 0.92, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=10, linespacing=1.7)

    ax = fig.add_subplot(gs[2, :])
    acc = imu[imu.sensor == "acc"]
    ta = (acc.timestamp - t0) / 1000
    ax.plot(ta, np.sqrt(acc.x**2 + acc.y**2 + acc.z**2), lw=0.6, color="#888", label="가속도 크기")
    for i, st in enumerate((step_t - t0) / 1000):
        ax.axvline(st, color="#0F7B5F", lw=0.6, alpha=0.6, label="검출 걸음" if i == 0 else None)
    for _, r in gt.iterrows():
        x = (r.timestamp - t0) / 1000
        ax.axvline(x, color="#D99A00", lw=2)
        ax.text(x, 1.02, f"핑 {int(r.seq)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9,
                color="#8A5A00", fontweight="bold")
    ax.plot((scans.timestamp - t0) / 1000, np.full(len(scans), 6.5), "s", color="#2F5BD3", ms=6,
            label="Wi-Fi 스캔")
    ax.set_xlabel("기록 시작 후 시간 (s)")
    ax.set_ylabel("m/s²")
    ax.set_ylim(5.5, None)
    ax.legend(loc="upper right", fontsize=8.5, ncol=3)
    ax.set_title("(d) 시간축: 핑 · 걸음 · Wi-Fi 스캔", loc="left", fontsize=11, pad=16)

    fig.suptitle(f"위치 정답(GT) 복원 — {name}", fontsize=14, fontweight="bold", x=0.06, ha="left", y=0.98)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gt.png", dpi=150, bbox_inches="tight")
    return fig


def run(data_dir: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT):
    """gt.csv 가 있는 세션마다 복원. {세션: 구간 표} 반환"""
    results = {}
    for d in sorted(p for p in Path(data_dir).iterdir() if (p / "gt.csv").exists()):
        meta, gt, imu, wifi = load(d)
        if len(gt) < 2:
            print(f"{d.name}: 핑이 2개 미만이라 건너뜀")
            continue
        track, raw, seg, step_t = reconstruct(gt, imu)
        wifi_lab = label_wifi(wifi, track, gt)
        out = Path(out_dir) / d.name
        out.mkdir(parents=True, exist_ok=True)
        track.round(3).to_csv(out / "gt_track.csv", index=False)
        wifi_lab.to_csv(out / "wifi_labeled.csv", index=False, encoding="utf-8-sig")
        seg.to_csv(out / "gt_segments.csv", index=False, encoding="utf-8-sig")
        fig = plot(d.name, d, meta, gt, imu, wifi_lab, track, raw, seg, step_t, out)
        plt.close(fig)
        results[d.name] = seg
        print(d.name)
        print(seg.to_string(index=False))
    return results


if __name__ == "__main__":
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    run(data, out)
