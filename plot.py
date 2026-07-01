import matplotlib.pyplot as plt
import numpy as np

def plot_scan(angles, ranges, intensivities, potential_peaks, trash, threshold=300):
    order = sorted(range(len(angles)), key=lambda i: angles[i])
    a = [angles[i] for i in order]
    r = [ranges[i] for i in order]
    inten = [intensivities[i] for i in order]

    # gap считаем ТАК ЖЕ, как фильтр: abs(r - r_prev), с обрывом на дропаутах
    gaps = [np.nan] * len(r)
    prev_r = None
    for i in range(len(r)):
        if r[i] <= 0:
            prev_r = None
            continue
        if prev_r is not None:
            gaps[i] = abs(r[i] - prev_r)
        prev_r = r[i]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Результаты скана")

    # --- 1: расстояние ---
    ax1.plot(a, r, color="tab:blue", linewidth=0.8, marker=".", markersize=3, label="дальности")
    ax1.set_title("Расстояние (ranges)")
    ax1.set_ylabel("Расстояние, мм")
    trash_x = [angles[i] for i in trash]
    trash_y = [ranges[i] for i in trash]
    
    ax1.plot(trash_x, trash_y, color="tab:orange", linestyle="", marker=".", markersize=5, label="мусор")

    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- 2: gap (разрыв дальности) ---
    ax2.plot(a, gaps, color="tab:gray", linewidth=0.6, marker=".", markersize=3)
    ax2.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.2, label=f"порог {threshold} мм")
    over = [i for i in range(len(gaps)) if not np.isnan(gaps[i]) and gaps[i] > threshold]
    ax2.plot([a[i] for i in over], [gaps[i] for i in over], "o", color="tab:red", markersize=6, label="разрыв (новый сегмент)")
    ax2.set_title("gap = |r - r_prev| — где красное, там скан режется")
    ax2.set_ylabel("gap, мм")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- 3: интенсивность + пики ---
    ax3.plot(a, inten, color="tab:red", linewidth=0.8, marker=".", markersize=3, label="интенсивность")
    ax3.set_title("Интенсивность (intensities) | зелёное — потенциальные пики")
    ax3.set_xlabel("Угол, градусы")
    ax3.set_ylabel("Интенсивность (quality)")
    ax3.set_xlim(0, 360)
    ax3.grid(True, alpha=0.3)

    # Флаг, чтобы добавить подпись в легенду только один раз
    label_added = False

    for seq in potential_peaks:
        idx = sorted((point.index for point in seq), key=lambda i: angles[i])

        # Передаем label только для первого куска, для остальных — None
        current_label = "пики интенсивности" if not label_added else None
        ax3.plot([angles[i] for i in idx], [intensivities[i] for i in idx], color="tab:green", linewidth=1.5, marker=".", markersize=4, label=current_label)
        label_added = True  # После первой итерации отключаем добавление label

    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
def plot_topdown(coordinates, ranges, potential_peaks=None, zoom=None):
    """Вид сверху (карта): точки скана в X/Y, лидар в центре,
    кандидаты выделены, и через них проведена прямая (np.polyfit).
 
    coordinates      : glass_filter.coordinates  ([[x, y], ...])
    ranges           : scan (чтобы понять, какие лучи валидные)
    potential_peaks  : список сегментов (списки Point); можно None
    zoom             : (xmin, xmax, ymin, ymax) или None
    """
    X = np.array([c[0] for c in coordinates])
    Y = np.array([c[1] for c in coordinates])
    valid = np.array(ranges) > 0
 
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Вид сверху")
 
    # фон: все валидные точки серым
    ax.scatter(X[valid], Y[valid], s=8, c="lightgray", zorder=1)
    # лидар в начале координат
    ax.scatter([0], [0], marker="*", s=300, c="black", zorder=6, label="лидар")
 
    colors = ["tab:red", "tab:purple", "tab:orange", "tab:green", "tab:brown"]
    if potential_peaks:
        for i, seg in enumerate(potential_peaks):
            c = colors[i % len(colors)]
            xs = np.array([p.x for p in seg])
            ys = np.array([p.y for p in seg])
 
            k, b = np.polyfit(xs, ys, 1)              # прямая-поверхность
            x0, x1 = xs.min(), xs.max()
            xext = np.linspace(x0 - 250, x1 + 250, 50)
            ax.plot(xext, k * xext + b, "--", color=c, lw=1.0, alpha=0.7)   # продолжение
            xin = np.linspace(x0, x1, 50)
            ax.plot(xin, k * xin + b, "-", color=c, lw=2.6, zorder=4)       # сама поверхность
 
            ax.scatter(xs, ys, s=55, c=c, edgecolors="k", linewidths=0.4, zorder=5)
            # границы (первая и последняя точка сегмента)
            ax.scatter([seg[0].x, seg[-1].x], [seg[0].y, seg[-1].y],
                       s=120, marker="s", facecolors="none", edgecolors=c,
                       linewidths=2, zorder=6)
            ax.annotate(f"{seg[0].index}-{seg[-1].index}", (xs.mean(), ys.mean()),
                        textcoords="offset points", xytext=(10, 10),
                        color=c, fontsize=10, fontweight="bold")
 
    ax.set_aspect("equal")            # <-- БЕЗ ЭТОГО комната «сплющена»
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X, мм")
    ax.set_ylabel("Y, мм")
    ax.legend(loc="lower right", fontsize=9)
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
 
    plt.tight_layout()
    plt.show()
 
 
# Пример вызова (у тебя glass_filter уже построен):
#
#   plot_topdown(glass_filter.coordinates, scan, potential_peaks)
#   plot_topdown(glass_filter.coordinates, scan, potential_peaks,
#                zoom=(-820, 260, -40, 470))
 
 
def plot_scan_comparison(scan_before, scan_after, occupied=None, surfaces=None, zoom=None):
    """Сравнение сканов до/после (вид сверху, две панели рядом).
 
    Args:
        scan_before : массив дальностей ДО фильтрации
        scan_after  : массив дальностей ПОСЛЕ (после patch_glass)
        occupied    : маска (True там, где точку поджали к стеклу); можно None
        surfaces    : список Surface с .p_start/.p_end для отрисовки стекла; можно None
        zoom        : (xmin, xmax, ymin, ymax) для приближения; можно None
    """
    N = len(scan_before)
    az = np.deg2rad(np.arange(N))
    Xb, Yb = np.array(scan_before) * np.cos(az), np.array(scan_before) * np.sin(az)
    Xa, Ya = np.array(scan_after) * np.cos(az),  np.array(scan_after) * np.sin(az)
    vb = np.array(scan_before) > 0
    va = np.array(scan_after) > 0
    occ = np.array(occupied, dtype=bool) if occupied is not None else np.zeros(N, bool)
 
    fig, (axB, axA) = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    fig.suptitle("Сравнение сканов: до / после", fontsize=13, fontweight="bold")
 
    for ax in (axB, axA):
        if surfaces:
            for s in surfaces:
                ax.plot([s.p_start.x, s.p_end.x], [s.p_start.y, s.p_end.y],
                        "-", color="tab:blue", lw=2.5, alpha=0.5, zorder=2)
        ax.scatter([0], [0], marker="*", s=280, c="black", zorder=6, label="лидар")
        ax.set_aspect("equal")            # без этого комната сплющена
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X, мм"); ax.set_ylabel("Y, мм")
        if zoom:
            ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
 
    axB.scatter(Xb[vb], Yb[vb], s=12, c="tab:orange", zorder=3, label="скан")
    axB.set_title("ДО"); axB.legend(loc="upper left", fontsize=9)
 
    axA.scatter(Xa[va & ~occ], Ya[va & ~occ], s=12, c="lightgray", zorder=3, label="норма")
    axA.scatter(Xa[va & occ],  Ya[va & occ],  s=26, c="tab:red",   zorder=4, label="поджато к стеклу")
    axA.set_title("ПОСЛЕ"); axA.legend(loc="upper left", fontsize=9)
 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
