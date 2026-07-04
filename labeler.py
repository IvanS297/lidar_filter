"""
Разметчик скана лидара для калибровки СКО-фильтра (Tibebu, Алгоритм 1).

Что делает:
- показывает скан по углу (дальность) + скользящее СКО + вид сверху;
- мышкой выделяешь диапазон углов и помечаешь его как СТЕКЛО или СТЕНА;
- считает медианы СКО двух групп и предлагает порог Trh = ½(среднее_стекло+среднее_стена);
- по кнопке сохраняет разметку в labels.json.

Запуск:
    python label_tool.py               # на встроенном тестовом скане
    python label_tool.py my_scan.py    # my_scan.py задаёт scan и intesivities (np.array)

Управление:
    - выбери режим слева (стекло / стена / стереть);
    - зажми левую кнопку мыши на ВЕРХНЕЙ панели и протяни по углам — диапазон разметится;
    - клавиши: 1=стекло, 2=стена, 0=стереть, s=сохранить, r=сброс;
    - кнопка «Сохранить» пишет labels.json и печатает предложенный порог.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, RadioButtons, Button


# ---------- скользящее СКО (Алгоритм 1) ----------
def rolling_stdv(scan, half=5):
    N = len(scan)
    sig = np.full(N, np.nan)
    for i in range(N):
        idx = [(i + k) % N for k in range(-half, half + 1)]   # круговое окно 11
        r = [scan[j] for j in idx if scan[j] > 0]             # без дропаутов
        if len(r) >= 3:
            mu = sum(r) / len(r)
            sig[i] = (sum((x - mu) ** 2 for x in r) / len(r)) ** 0.5
    return sig


# ---------- загрузка данных ----------
def load_scan(path):
    ns = {"np": np}
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), ns)
    scan = np.asarray(ns["scan"], dtype=float)
    inten = ns.get("intesivities", ns.get("inten", None))
    inten = np.asarray(inten, dtype=float) if inten is not None else np.zeros_like(scan)
    return scan, inten


def make_demo():
    """Встроенный тестовый скан: стена + сквозное окно (чтобы было что размечать)."""
    N = 360
    scan = np.zeros(N)
    for i in range(N):
        th = np.deg2rad(i)
        dx = np.cos(th)
        if dx > 0.05:
            gy = (2000.0 / dx) * np.sin(th)
            if -700 <= gy <= 700:            # сектор окна -> видим стену за ним (сквозняк)
                scan[i] = 3000.0 / dx + np.random.uniform(-200, 200)
            else:
                scan[i] = 1500.0 / dx
        else:
            scan[i] = 1500.0
    inten = np.where(scan > 2500, 15, 45).astype(float)
    return scan, inten


if len(sys.argv) > 1:
    scan, inten = load_scan(sys.argv[1])
else:
    print("Файл скана не задан — открываю встроенный тестовый скан.")
    scan, inten = make_demo()

N = len(scan)
sig = rolling_stdv(scan)
labels = np.zeros(N, dtype=int)         # 0=не размечено, 1=стекло, 2=стена
ang = np.arange(N)
az = np.deg2rad(ang)
X = scan * np.cos(az)
Y = scan * np.sin(az)

LBL_COLORS = {0: "lightgray", 1: "tab:orange", 2: "tab:blue"}
mode = {"cur": 1}                        # текущий режим разметки


# ---------- фигура ----------
fig, (axR, axS, axT) = plt.subplots(3, 1, figsize=(12, 9))
plt.subplots_adjust(left=0.26, bottom=0.06, top=0.93, hspace=0.45)


def suggested_thr():
    gv = sig[(labels == 1) & ~np.isnan(sig)]
    wv = sig[(labels == 2) & ~np.isnan(sig)]
    if len(gv) and len(wv):
        return 0.5 * (np.mean(gv) + np.mean(wv)), np.median(gv), np.median(wv)
    return None, (np.median(gv) if len(gv) else None), (np.median(wv) if len(wv) else None)


def redraw():
    thr, gmed, wmed = suggested_thr()

    # --- панель 1: дальность + подсветка разметки ---
    axR.clear()
    v = scan > 0
    axR.plot(ang[v], scan[v], ".-", color="tab:gray", lw=0.5, ms=3)
    for lab, col in ((1, "tab:orange"), (2, "tab:blue")):
        m = labels == lab
        axR.scatter(ang[m & v], scan[m & v], s=25, c=col, zorder=5)
    axR.set_title("Дальность по углу  —  тяни мышкой здесь, чтобы разметить")
    axR.set_ylabel("дальность, мм")
    axR.grid(True, alpha=0.3)
    axR.set_xlim(0, N)

    # --- панель 2: СКО + порог ---
    axS.clear()
    axS.plot(ang, sig, ".-", color="tab:green", lw=0.6, ms=3)
    if thr is not None:
        axS.axhline(thr, color="tab:red", ls="--", lw=1.3, label="порог Trh=%.0f" % thr)
        axS.legend(loc="upper right", fontsize=9)
    axS.set_title("Скользящее СКО (окно 11)")
    axS.set_ylabel("СКО, мм")
    axS.grid(True, alpha=0.3)
    axS.set_xlim(0, N)

    # --- панель 3: вид сверху ---
    axT.clear()
    for lab, col in ((0, "lightgray"), (2, "tab:blue"), (1, "tab:orange")):
        m = (labels == lab) & (scan > 0)
        axT.scatter(X[m], Y[m], s=10, c=col, zorder=(3 if lab else 1))
    axT.scatter([0], [0], marker="*", s=200, c="black", zorder=6)
    axT.set_aspect("equal")
    axT.grid(True, alpha=0.3)
    axT.set_title("Вид сверху (оранжевое=стекло, синее=стена)")
    axT.set_xlabel("X, мм")

    # заголовок с текущими цифрами
    parts = ["режим: " + {1: "СТЕКЛО", 2: "СТЕНА", 0: "СТЕРЕТЬ"}[mode["cur"]]]
    if gmed is not None:
        parts.append("СКО стекла≈%.0f" % gmed)
    if wmed is not None:
        parts.append("СКО стены≈%.0f" % wmed)
    if thr is not None:
        parts.append("порог≈%.0f" % thr)
    fig.suptitle("   |   ".join(parts), fontsize=11)
    fig.canvas.draw_idle()


# ---------- виджеты ----------
ax_radio = plt.axes([0.02, 0.62, 0.2, 0.22])
radio = RadioButtons(ax_radio, ("стекло (PCoG)", "стена (PCoO)", "стереть"))
_mode_map = {"стекло (PCoG)": 1, "стена (PCoO)": 2, "стереть": 0}


def on_radio(label):
    mode["cur"] = _mode_map[label]
    redraw()


radio.on_clicked(on_radio)


def onselect(xmin, xmax):
    a, b = int(round(xmin)), int(round(xmax))
    a, b = max(0, a), min(N - 1, b)
    labels[a:b + 1] = mode["cur"]
    redraw()


try:
    span = SpanSelector(axR, onselect, "horizontal", useblit=True,
                        props=dict(alpha=0.2, facecolor="tab:gray"))
except TypeError:   # старые версии matplotlib
    span = SpanSelector(axR, onselect, "horizontal", useblit=True,
                        rectprops=dict(alpha=0.2, facecolor="tab:gray"))


def do_save(event=None):
    glass = [int(i) for i in range(N) if labels[i] == 1]
    wall = [int(i) for i in range(N) if labels[i] == 2]
    thr, gmed, wmed = suggested_thr()
    out = {"glass": glass, "wall": wall,
           "Trh": (float(thr) if thr is not None else None),
           "stdv_median_glass": (float(gmed) if gmed is not None else None),
           "stdv_median_wall": (float(wmed) if wmed is not None else None)}
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Сохранено labels.json | стекло=%d лучей, стена=%d лучей, порог Trh=%s"
          % (len(glass), len(wall), ("%.0f" % thr) if thr else "—"))


def do_reset(event=None):
    labels[:] = 0
    redraw()


ax_save = plt.axes([0.02, 0.50, 0.2, 0.07])
Button(ax_save, "Сохранить").on_clicked(do_save)
ax_reset = plt.axes([0.02, 0.41, 0.2, 0.07])
Button(ax_reset, "Сброс").on_clicked(do_reset)


def on_key(event):
    if event.key in ("1", "2", "0"):
        mode["cur"] = int(event.key)
        redraw()
    elif event.key == "s":
        do_save()
    elif event.key == "r":
        do_reset()


fig.canvas.mpl_connect("key_press_event", on_key)

# кнопки Save/Reset нужно держать в переменных, иначе GC их съест
_keep = (radio, span)

redraw()
plt.show()