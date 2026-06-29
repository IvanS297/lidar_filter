import numpy as np
import scipy
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x: float, y: float, intensivity: float = None, index: int = None, range: int = None, angle: int = None):
        """Объект точки со скана лидара

        Args:
            x (float): x точки
            y (float): y точки
            intensivity (float, optional): Интенсивность, которая дана точке. Defaults to None.
            index (int, optional): Индекс в массиве скана. Defaults to None.
            range (int, optional): Значение дальности до объекта. Defaults to None.
            angle (int, optional): Угол 0..359 скана лидара. Defaults to None.
        """
        self.x = x
        self.y = y
        self.intensivity = intensivity
        self.index = index
        self.range = range
        self.deg_angle = angle

    def set_coordinates(self, x: float, y: float):
        """Изменить x y точки

        Args:
            x (float): новый x
            y (float): новый y
        """
        self.x = x
        self.y = y

    def set_intensivity(self, intensivity: float):
        """Изменить интенсивность точки

        Args:
            intensivity (float): новая интенсивность
        """
        self.intensivity = intensivity

    def set_index(self, index: int):
        """Поменять индекс точки

        Args:
            index (int): новый индекс в массиве скана
        """
        self.index = index

    def set_range(self, range: int):
        """Поменять дальность до объекта в мм

        Args:
            range (int): Новое расстояние в мм
        """
        self.range = range

    def set_angle(self, angle: int):
        """Изменить угол в градусах для данной точки

        Args:
            angle (int): угол от 0 до 359
        """
        self.deg_angle = angle


class GlassFilter:
    def __init__(self, max_cliff: int = 300):
        """Фильтр, который выявляет куски скана, похожие на пик интенсивности. Эти пики интенсивности появляются из-за встречи луча лидара и стекла или зеркала.

        Args:
            max_cliff (int, optional): Максимальный разрыв по дальности между точками в мм. Defaults to 300.
        """
        self.max_cliff = max_cliff
        self.sequence = []
        self.coordinates = []

    def scan_to_points(self, x: float, y: float, theta: float, scan: np.array) -> list:
        """Находим координаты препятсвий по скану

        Args:
            x (float): x робота
            y (float): y робота
            theta (float): theta робота
            scan (np.array): скан

        Returns:
            list: массив с массивами координат x y перпятсвий
        """
        coordinates = []  # [[x, y], [x1, y1]]
        for i in range(len(scan)):
            xo = x + scan[i] * np.cos(theta + np.deg2rad(i))
            yo = y + scan[i] * np.sin(theta + np.deg2rad(i))
            coordinates.append([xo, yo])
        return coordinates

    def is_seq_is_valid(self, min_points: int, min_amp: int, x: float, y: float, max_chng: int) -> bool:
        """Проверка, что sequence с point'ами вообще правильна, на основе ее физических свойств

        Args:
            min_points (int): минимальный размер sequence
            min_amp (int): минимальная амплитуда скачков
            x (float): x робота
            y (float): y робота
            max_chng (int): максимальное отколенене в дальности между точками и лидаром в мм

        Returns:
            valid (bool): True если все 4 условия существования такой sequence верны, False - если хотя бы одно не выполнилось.
        """

        """
        форма пика: интенсивность точек в sequence  должна сначала монотонно возрастать, а затем убывать
        минимальная ширина: т.к. пик интенсивности это практически всегда мусор, то длина последовательности должна быть маленькой
        амплитуда: разница между максимальной интенсивностью и минимальной должна быть больше заданного порога
        физическая непрерывность: расстояние между лидаром и точками sequene не должна резко меняться.
        """
        # ширина
        # фильтруем сначала по ширине последовательности, потому что дальше есть функции, которые работают с итераторами в списках. МОжет быть IndexErro или какой-нибудь дргуой краш
        len_valid = len(self.sequence) >= min_points
        if not len_valid:
            return False

        # амплитуда:
        max_ind, i_max = max(enumerate(self.sequence[i].intensivity for i in range(
            len(self.sequence))), key=lambda x: x[1])
        """
        Есть такой неприятный баг: вершина может оказаться вообще на краю последовательности (индекс 0 или индекс len(sequence) - 1) и тогда логика ломается.
        """
        if not (0 < max_ind < len(self.sequence) - 1):
            return False
        i_edge = (self.sequence[0].intensivity +
                  self.sequence[-1].intensivity) / 2
        amp_valid = (i_max - i_edge) > min_amp
        if not amp_valid:
            return False

        # непрерывность
        d = prevd = dd = 0
        cont_valid = True
        for i in range(0, len(self.sequence)):
            point = self.sequence[i]
            d = np.sqrt(np.pow(point.x - x, 2) + np.pow(point.y - y, 2))
            if i == 0:
                prevd = d
                continue
            dd = prevd - d
            if dd > max_chng:
                cont_valid = False
            prevd = d
        if not cont_valid:
            return False

        # возрастание и убывание
        shape_valid = True
        for i in range(1, max_ind + 1):
            if self.sequence[i - 1].intensivity > self.sequence[i].intensivity:
                shape_valid = False

        for i in range(max_ind + 1, len(self.sequence)):
            if self.sequence[i - 1].intensivity < self.sequence[i].intensivity:
                shape_valid = False

        if not shape_valid:
            return False

        return True

    def find_potential_peaks(self, intensiv: list, scan: list) -> list:
        potential_peaks = []
        self.sequence = []  # последовательность, которая накапливается point'ами в процессе
        prev_r = None
        for r in range(len(scan)):
            # дропаут (нулевая интенсивность или проще говоря мусор, разрыв графика)
            if scan[r] <= 0:
                if self.is_seq_is_valid(min_points=3, min_amp=3, x=0, y=0, max_chng=300):
                    potential_peaks.append(self.sequence)
                self.sequence = []
                prev_r = None
                continue

            # gap - это физический разрыв между точками скана (длина между ними). Проще говоря это евклидово расстояние между соседними точками
            gap = abs(scan[r] - prev_r) if prev_r is not None else 0.0

            if gap > self.max_cliff:  # разрыв дальности означает, что сегмент кончился
                if self.is_seq_is_valid(min_points=3, min_amp=3, x=0, y=0, max_chng=300):
                    # сохранение последовательности точек (пик) в массив потенциальных пиков
                    potential_peaks.append(self.sequence)
                self.sequence = []
            point = Point(x=self.coordinates[r][0], y=self.coordinates[r][1],
                          intensivity=intensiv[r], index=r, range=scan[r], angle=r)
            self.sequence.append(point)
            prev_r = scan[r]

        # последний кусок
        if self.is_seq_is_valid(min_points=3, min_amp=3, x=0, y=0, max_chng=300):
            potential_peaks.append(self.sequence)
        return potential_peaks


def plot_scan(angles, ranges, intensities, potential_peaks, threshold=300):
    order = sorted(range(len(angles)), key=lambda i: angles[i])
    a = [angles[i] for i in order]
    r = [ranges[i] for i in order]
    inten = [intensities[i] for i in order]

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
    ax1.plot(a, r, color="tab:blue", linewidth=0.8, marker=".", markersize=3)
    ax1.set_title("Расстояние (ranges)")
    ax1.set_ylabel("Расстояние, мм")
    ax1.grid(True, alpha=0.3)

    # --- 2: gap (разрыв дальности) ---
    ax2.plot(a, gaps, color="tab:gray",
             linewidth=0.6, marker=".", markersize=3)
    ax2.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.2,
                label=f"порог {threshold} мм")
    over = [i for i in range(len(gaps)) if not np.isnan(
        gaps[i]) and gaps[i] > threshold]
    ax2.plot([a[i] for i in over], [gaps[i] for i in over], "o",
             color="tab:red", markersize=6, label="разрыв (новый сегмент)")
    ax2.set_title("gap = |r - r_prev| — где красное, там скан режется")
    ax2.set_ylabel("gap, мм")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- 3: интенсивность + пики ---
    ax3.plot(a, inten, color="tab:red",
             linewidth=0.8, marker=".", markersize=3, label="интенсивность")
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

        ax3.plot([angles[i] for i in idx], [intensities[i] for i in idx],
                 color="tab:green", linewidth=1.5, marker=".", markersize=4,
                 label=current_label)

        label_added = True  # После первой итерации отключаем добавление label

    # ОБЯЗАТЕЛЬНО: включаем отображение легенды для третьего графика
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


scan = np.array([370.5, 368.5, 366.25, 363.25, 362.25, 358.75, 359.25, 353.5, 353.75, 352.0, 0.0, 256.25, 253.5, 255.75, 0.0, 326.5, 257.5, 256.75, 256.25, 258.0, 259.0, 340.75, 342.0, 341.5, 341.0, 339.5, 339.75, 341.25, 340.5, 342.0, 341.5, 343.25, 339.5, 339.25, 343.5, 346.5, 347.0, 347.75, 349.0, 352.0, 352.5, 354.5, 356.0, 359.0, 361.25, 361.25, 364.0, 366.25, 368.5, 370.0, 375.0, 377.25, 381.25, 383.25, 388.25, 391.25, 395.5, 401.0, 403.5, 410.0, 413.0, 418.75, 422.25, 430.5, 439.0, 442.5, 446.25, 455.5, 464.75, 0.0, 269.5, 266.5, 262.25, 258.75, 263.5, 265.5, 272.5, 274.25, 273.25, 0.0, 0.0, 576.0, 572.5, 0.0, 0.0, 351.25, 354.25, 358.75, 362.5, 361.25, 0.0, 0.0, 377.75, 376.75, 0.0, 0.0, 244.75, 246.75, 256.25, 260.0, 268.25, 268.0, 272.0, 278.5, 281.25, 286.25, 291.75, 295.0, 299.75, 303.75, 313.5, 321.25, 325.25, 329.25, 335.5, 346.75, 352.5, 361.75, 368.0, 380.25, 388.75, 394.0, 413.5, 428.0, 448.25, 458.5, 480.0, 491.25, 496.0, 836.75, 853.0, 806.0, 801.0, 806.0, 810.25, 813.75, 818.5, 820.75, 825.0, 834.5, 844.75, 849.25, 853.5, 862.5, 877.0, 874.25, 806.5, 805.0, 810.75, 813.25, 823.0, 825.0, 816.0, 0.0, 716.75, 704.25, 706.75, 713.5, 717.5, 736.5, 750.5, 0.0, 922.25, 735.5, 723.5, 721.25, 725.75, 728.0, 733.0, 749.25, 761.75, 1077.0, 0.0, 1182.0, 403.5, 389.5, 384.5,
                374.5, 371.5, 366.5, 365.75, 364.0, 364.5, 366.0, 390.0, 409.5, 415.25, 1968.0, 1981.5, 0.0, 0.0, 2405.5, 2385.5, 0.0, 491.0, 0.0, 401.0, 397.75, 396.0, 396.5, 397.5, 399.5, 393.5, 385.75, 377.5, 376.0, 380.5, 382.0, 384.25, 386.75, 387.0, 385.75, 384.0, 384.0, 379.25, 379.0, 382.0, 383.0, 389.5, 391.5, 397.25, 400.0, 403.0, 405.0, 413.5, 418.0, 431.0, 436.0, 448.5, 446.5, 441.0, 135.5, 136.0, 137.25, 138.75, 138.5, 138.75, 140.0, 143.0, 146.0, 147.0, 148.5, 151.0, 154.0, 154.75, 154.75, 155.5, 157.0, 158.0, 160.25, 0.0, 0.0, 0.0, 3859.5, 3814.0, 3793.0, 3774.0, 3718.0, 3707.0, 3739.5, 3767.0, 0.0, 0.0, 0.0, 4352.25, 4399.5, 4439.0, 4429.5, 4299.0, 4285.0, 4285.0, 3841.25, 3911.0, 3962.5, 4072.0, 4128.0, 4238.75, 0.0, 5495.5, 5459.5, 5413.0, 5408.0, 5392.75, 0.0, 3734.0, 3676.75, 0.0, 459.25, 430.0, 0.0, 0.0, 347.75, 0.0, 312.75, 305.0, 300.5, 0.0, 0.0, 257.75, 247.0, 0.0, 0.0, 0.0, 215.0, 209.25, 198.0, 0.0, 200.0, 182.75, 182.0, 0.0, 159.75, 150.5, 151.0, 0.0, 361.0, 366.75, 369.75, 373.5, 377.25, 379.0, 380.75, 386.75, 389.5, 391.25, 397.5, 400.5, 407.25, 410.0, 416.25, 421.75, 428.75, 559.0, 550.75, 0.0, 532.0, 0.0, 501.75, 492.5, 494.5, 234.0, 0.0, 457.75, 454.75, 446.25, 438.5, 435.75, 427.5, 422.25, 0.0, 0.0, 402.0, 403.0, 397.5, 392.25, 388.75, 386.75, 383.5, 377.75, 374.25])
intesivities = np.array([52, 52, 52, 53, 53, 53, 53, 53, 53, 52, 0.0, 36, 37, 37, 0.0, 49, 52, 49, 39, 37, 34, 54, 53, 53, 54, 52, 53, 53, 53, 53, 53, 52, 22, 20, 52, 52, 53, 53, 53, 52, 52, 52, 52, 52, 52, 52, 52, 52, 51, 51, 52, 52, 52, 51, 52, 51, 51, 51, 51, 50, 50, 49, 50, 50, 50, 49, 49, 48, 44, 0.0, 50, 51, 53, 53, 51, 51, 50, 48, 35, 0.0, 0.0, 40, 42, 0.0, 0.0, 47, 47, 47, 48, 26, 0.0, 0.0, 46, 42, 0.0, 0.0, 17, 22, 24, 26, 25, 24, 25, 26, 24, 26, 24, 24, 24, 24, 23, 23, 23, 22, 22, 20, 21, 20, 20, 20, 20, 20, 19, 18, 16, 15, 14, 14, 19, 47, 52, 46, 46, 46, 46, 44, 45, 44, 43, 43, 43, 43, 43, 43, 44, 42, 42, 50, 44, 47, 48, 48, 46, 0.0, 37, 47, 47, 45, 43, 43, 39, 0.0, 46, 39, 40, 51, 49, 45, 44, 42, 33, 36, 0.0, 39, 37, 43, 44,
                        44, 46, 46, 47, 45, 45, 42, 45, 45, 36, 35, 36, 0.0, 0.0, 34, 28, 0.0, 44, 0.0, 50, 50, 50, 50, 51, 52, 51, 50, 50, 50, 50, 50, 51, 52, 51, 51, 51, 50, 49, 49, 50, 50, 50, 50, 50, 50, 50, 49, 49, 48, 48, 48, 48, 40, 22, 44, 51, 53, 55, 55, 54, 55, 52, 49, 46, 48, 50, 53, 53, 52, 52, 51, 49, 44, 0.0, 0.0, 0.0, 26, 32, 32, 32, 30, 32, 31, 11, 0.0, 0.0, 0.0, 21, 24, 1, 2, 18, 18, 1, 21, 19, 20, 19, 19, 13, 0.0, 21, 18, 19, 21, 17, 0.0, 29, 32, 0.0, 15, 11, 0.0, 0.0, 10, 0.0, 8, 9, 6, 0.0, 0.0, 7, 5, 0.0, 0.0, 0.0, 8, 6, 6, 0.0, 7, 7, 6, 0.0, 5, 8, 5, 0.0, 42, 42, 42, 41, 42, 42, 42, 42, 42, 43, 43, 43, 42, 42, 42, 38, 32, 45, 28, 0.0, 47, 0.0, 49, 30, 7, 5, 0.0, 49, 51, 50, 50, 51, 51, 47, 0.0, 0.0, 51, 51, 51, 52, 52, 52, 52, 52, 53])
angles = np.array([i for i in range(0, 360)])
glass_filter = GlassFilter(300)
glass_filter.coordinates = glass_filter.scan_to_points(0, 0, 0, scan)
potential_peaks = glass_filter.find_potential_peaks(
    intensiv=intesivities, scan=scan)
for seq in potential_peaks:
    for point in seq:
        print(
            f"index: {point.index} x: {point.x} y: {point.y} angle: {point.deg_angle} intensivity: {point.intensivity}")
plot_scan(angles, scan, intesivities, potential_peaks)
