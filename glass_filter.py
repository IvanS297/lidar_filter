import numpy as np
from PCA import pca
import matplotlib.pyplot as plt
from plot import *
# https://pmc.ncbi.nlm.nih.gov/articles/PMC11314935/

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


class Surface:
    def __init__(self, k: float, b: float, beams: list, p_start: Point, p_end: Point):
        """Объект данных для хранения информации о поверхности, которая пердставляет собой в 2д пространстве простую прямую.
        Поверхность существует только в пределах скана лидара (0-359 градусов). Под поверхностью подразумевается 2д срез 3д объекта, который
        является стетлом, зеркалом ии другой отражающей поверхностью.

        Args:
            k (float): коэффициент k прямой поверхности
            b (float): коэффициент b прямой поверхности
            beams (list): массив лучей лидара, которые попадают в эту поверхность
            p_start (Point): начало поверхности
            p_end (Point): конец поверхности
        """
        self.k = k
        self.b = b
        self.beams = beams
        self.p_start = p_start
        self.p_end = p_end    
    
    def set_coeffs(self, k: float = None, b: float = None):
        """Поменять коэффициенты b и k на новые

        Args:
            k (float, optional): новый коэффициент K. Defaults to None.
            b (float, optional): новый коэффициент B. Defaults to None.
        """
        if k != None:
            self.k = k
        if b != None:
            self.b = b
        
    def set_point(self, ps: Point = None, pe: Point = None):
        """Поменять начало или конец пика

        Args:
            ps (Point, optional): Новый старт. Defaults to None.
            pe (Point, optional): Новый конец. Defaults to None.
        """
        if ps != None:
            self.p_start = ps
        if pe != None:    
            self.p_end = pe
    
    def set_beams(self, beams: list):
        """Поменять массив лучей поверхности

        Args:
            beams (list): массив лучей
        """
        self.beams = beams
    

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
        max_ind, i_max = max(enumerate(self.sequence[i].intensivity for i in range(len(self.sequence))), key=lambda x: x[1])
        """
        Есть такой неприятный баг: вершина может оказаться вообще на краю последовательности (индекс 0 или индекс len(sequence) - 1) и тогда логика ломается.
        """
        if not (0 < max_ind < len(self.sequence) - 1):
            return False
        i_edge = (self.sequence[0].intensivity + self.sequence[-1].intensivity) / 2
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

    def find_potential_peaks(self, intensiv: list, scan: list) -> list[list[Point]]:
        """Функция для нахождения потенциальных пиков и gap'ов из массива интенсивностей и массива измерений дальностей.
        Сама фильтрует gap'ы по пикам интенсивности, выдавая только те, которые дествительно похожи на отражающие поверности.

        Args:
            intensiv (list): массив интенсивностей
            scan (list): массив интенсивностей

        Returns:
            list[list[Point]]: отфильтрованный список пиков.
        """
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
            point = Point(x=self.coordinates[r][0], y=self.coordinates[r][1], intensivity=intensiv[r], index=r, range=scan[r], angle=r)
            self.sequence.append(point)
            prev_r = scan[r]

        # последний кусок
        if self.is_seq_is_valid(min_points=3, min_amp=3, x=0, y=0, max_chng=300):
            potential_peaks.append(self.sequence)
        return potential_peaks
    
    def fitting(self, potential_peaks) -> list[Surface]:
        """Функция, для фитинга прямой на кондидаты в отражающие поверхности. 

        Args:
            potential_peaks (list): потенциальные пики

        Returns:
            list[Surface]: массив поверхнстей
        """
        
        """
        Смысл в том, что эти potential_peaks и массивы с точками, которые показались алгоритму точками отражающей поверхности.
        Идея вот в чем: эти точки находятся рядом друг с дргуом. Лидар 2d пространства. Значит, чтобызадетектить в этих массивах зеркало или стекло, нужно проверить эти точки
        через уравнение прямой y = kx+b или ax+by+c=0. Если они "фитятся", то мы опнимаем, что это отражающая поверхность.
        В 3д пространстве зеркало и стекло - это большие поверхности (куски), а в 2д пространстве это прямая (срез 3д пространства).
        Но прямая бесконечная, а отражающая поверность - нет. поэтому надо было создать отдельный объект для поверхности.
        """
        
        surfaces = []
        for peak in potential_peaks:
            xs = np.array([p.x for p in peak]) # x точки пика
            ys = np.array([p.y for p in peak]) # y точки пика
            #k, b = np.polyfit(x=xs, y=ys, deg=1) # пытаемся получить k, b из уравнения y = kx+b
            # PCA вместо np.polyfit (в gitlog.txt написал почему)
            points = np.column_stack((xs, ys))
            _, _, _, _, k, b = pca(x=points, num_components=1, return_kb=True)
            start = Point(x=peak[0].x, y=peak[0].y) # Точка начала поверхности
            end = Point(x=peak[-1].x, y=peak[-1].y) # Точка конца поверхности
            beams = [p.index for p in peak] # какие лучи в него попали (индексы лучей)
            surface = Surface(k=k, b=b, p_start=start, p_end=end, beams=beams)
            surfaces.append(surface)
        
        return surfaces
    
    def find_real_trash(self, surfaces: list[Surface], scan: list[float], thresh: int = 35):
        """
        Функция берет каждый луч и смотрит, как луч идет до поверхности:
        1. Точка от луча оказалась перед стеклом?
        2. Точка на стекле?
        3. Точка за стеклом?
        
        Args:
            surfaces (list[Surface]): отражащие поверхности
            scan (list[float]): скан лидара
            thresh (int): сколько мм по обе стороны от зеркала считать той точкой, которая "лежит на зеркале"

        Returns:
            list[int]: Массив с индексами дальностей, которые стали мусорными.
        """
        
        """
        Описание геометрии.
        Луч номер i - это направление под углом theta_i. Точка вида (t*cos(theta_i); t*sin(theta_i)), где t >= 0 - расстояние вдоль луча (range).
        Стекло: Прямая в общей форме ax+by+c=0. У нее есть границы - отрезок начиная с p_start и заканчивая p_end.
        Пересечение: Надо решить уравнение, подставив точку луча в уравнение прямой. Решать надо относительно t:
        t = -c / (a*cos(theta_i) + b*sin(theta_i))
        Три варианта ответа при решении относительно t:
        1. если знаменатель ~= 0, то луч параллелен стеклу и не пересекает его.
        2. если t <= 0, то пересечение позади лидара, его не вопринимаем.
        3. иначе: точка пересечения = (t*cos(theta_i); t*sin(theta_i)). Надо проверить, что она лежит между p_start и p_end (действиельная точка, а не мнимая - точка на воображаемом продолжении прямой).
        Если нет - этот луч мимо стекла, не вопринимаем
        
        Если r ~= t, то луч утолкнулся в само стекло
        Если r > t, то луч прошел за стекло (может быть даже на сквозь), лиюо отразился. Это и есть мусор, который надо будет замять.
        Если r < t, то точка перед стеклом, обычное, препятствие. Точка в норме, не вопринимаем
        """
        new_scan = list(scan)                 # КОПИЯ, не ссылка
        occupied = [False] * len(scan)
        trash = []
        for i in range(len(scan)):            # лучи снаружи
            if scan[i] <= 0:
                continue
            theta = np.deg2rad(i)
            dx, dy = np.cos(theta), np.sin(theta)
            best_t = None                     # своё для КАЖДОГО луча
            for surface in surfaces:
                a, b0, c = surface.k, -1.0, surface.b
                x1, y1 = surface.p_start.x, surface.p_start.y
                x2, y2 = surface.p_end.x,   surface.p_end.y
                den = a * dx + b0 * dy
                if abs(den) < 1e-9:           # abs! иначе теряешь отрицательные
                    continue
                t = -c / den
                if t <= 0:
                    continue
                ix, iy = t * dx, t * dy
                Lx, Ly = x2 - x1, y2 - y1
                s = ((ix - x1) * Lx + (iy - y1) * Ly) / (Lx * Lx + Ly * Ly)
                if not (0.0 <= s <= 1.0):
                    continue
                if scan[i] - t >= thresh:              # луч за стеклом
                    if best_t is None or t < best_t:   # ближайшее стекло ДЛЯ ЭТОГО луча
                        best_t = t
            if best_t is not None:
                new_scan[i] = best_t
                occupied[i] = True
                trash.append(i)
        return new_scan, trash, occupied


scan = np.array([642.0, 643.0, 648.5, 3192.0, 3197.0, 3236.0, 0.0, 3350.0, 3342.0, 3345.0, 3330.0, 3320.5, 3382.0, 3388.5, 3402.5, 3410.0, 3426.5, 3443.0, 3445.25, 3452.0, 3404.0, 3168.0, 2974.5, 2920.0, 4819.0, 4836.5, 5922.5, 5979.0, 6003.5, 3633.0, 3656.0, 0.0, 3785.5, 0.0, 5370.0, 3912.5, 3970.5, 4034.5, 4067.0, 4131.0, 4164.0, 4235.5, 4288.5, 4250.0, 4140.5, 4088.0, 3990.0, 3944.5, 3856.0, 3815.5, 3733.5, 3655.5, 3620.0, 3550.5, 3519.5, 3454.5, 3399.0, 3373.5, 3318.5, 3292.5, 3243.5, 3221.0, 3178.5, 3139.5, 3119.5, 3097.0, 3061.5, 3044.5, 3010.0, 2976.0, 2961.0, 2935.5, 2921.0, 2895.0, 2887.0, 511.0, 503.5, 502.5, 502.0, 501.75, 506.0, 508.5, 513.0, 0.0, 363.5, 351.5, 346.0, 339.5, 336.0, 330.0, 326.5, 315.75, 311.25, 304.5, 301.0, 299.5, 294.5, 293.75, 290.5, 290.5, 290.5, 289.75, 288.5, 289.0, 288.0, 287.5, 288.5, 288.75, 289.5, 289.5, 291.5, 298.0, 299.25, 301.25, 304.25, 312.25, 315.25, 323.5, 327.25, 329.0, 334.5, 338.0, 345.75, 349.75, 359.5, 367.0, 0.0, 2330.0, 0.0, 377.0, 403.5, 406.5, 404.5, 0.0, 0.0, 353.0, 350.0, 331.25, 336.5, 0.0, 317.0, 301.5, 304.0, 302.5, 290.5, 285.0, 283.5, 285.0, 278.0, 279.75, 277.5, 273.0, 273.5, 269.25, 264.0, 262.5, 254.75, 256.0, 0.0, 246.5, 243.0, 240.0, 239.0, 236.5, 235.5, 236.25, 233.75, 234.5, 231.5, 232.0, 231.0, 231.0, 229.0, 231.0, 235.25, 236.5, 236.5, 238.5, 238.5, 241.5, 243.5, 244.5, 246.5, 248.5, 254.0, 248.0, 0.0, 246.5, 248.0, 252.5, 252.75, 253.75, 253.25, 0.0, 239.5, 234.0, 0.0, 0.0, 1395.5, 1407.0, 1389.0, 1365.0, 1367.0, 1343.0, 1255.0, 0.0, 1041.0, 1039.0, 1039.0, 1050.25, 1063.0, 1069.0, 1073.5, 1091.0, 1099.0, 1113.0, 1121.0, 1138.5, 1155.5, 1167.5, 1173.5, 0.0, 1222.5, 1220.0, 1183.5, 0.0, 1094.5, 1081.0, 1052.5, 1040.5, 1018.0, 1006.0, 984.5, 974.5, 954.25, 937.5, 929.0, 913.5, 906.0, 893.0, 886.5, 876.0, 0.0, 582.5, 591.5, 618.0, 632.0, 643.0, 645.0, 661.5, 673.75, 681.5, 684.5, 686.0, 0.0, 761.5, 772.0, 768.75, 765.5, 761.5, 759.5, 757.0, 754.5, 750.5, 749.25, 744.5, 744.0, 743.0, 742.0, 738.5, 737.0, 737.0, 735.5, 736.75, 735.75, 735.75, 735.75, 739.75, 740.5, 741.5, 742.0, 745.25, 747.0, 752.5, 0.0, 789.5, 793.0, 798.25, 800.25, 804.0, 806.5, 813.5, 810.5, 0.0, 0.0, 702.5, 661.5, 626.5, 609.0, 579.0, 565.5, 555.5, 533.5, 510.5, 501.5, 494.0, 0.0, 0.0, 614.5, 616.5, 625.0, 632.5, 0.0, 869.5, 871.0, 0.0, 757.0, 0.0, 858.5, 877.0, 877.0, 859.5, 862.5, 0.0, 827.75, 818.5, 809.0, 791.5, 784.5, 771.0, 764.0, 744.0, 738.0, 727.5, 724.0, 731.5, 0.0, 895.0, 919.0, 945.5, 1007.5, 0.0, 1081.5, 0.0, 1124.0, 1117.5, 1106.0, 1101.5, 1090.5, 1086.5, 1080.5, 1078.0, 1069.75, 1071.5, 1071.5, 673.5, 666.5, 652.0, 644.0, 641.5])
intesivities = np.array([41, 41, 17, 40, 19, 19, 0.0, 29, 13, 31, 20, 26, 30, 30, 29, 28, 29, 30, 28, 14, 25, 23, 14, 25, 19, 17, 13, 12, 12, 22, 18, 0.0, 18, 0.0, 2, 17, 21, 23, 21, 21, 21, 21, 22, 19, 21, 21, 21, 22, 23, 23, 23, 24, 24, 25, 25, 25, 26, 27, 25, 26, 26, 26, 28, 28, 28, 28, 28, 28, 27, 28, 28, 28, 28, 29, 20, 35, 40, 41, 42, 42, 40, 39, 34, 0.0, 34, 39, 40, 43, 43, 43, 42, 42, 43, 45, 46, 47, 47, 47, 48, 48, 48, 48, 49, 48, 48, 48, 48, 48, 48, 49, 48, 48, 47, 47, 46, 44, 44, 45, 45, 46, 45, 44, 43, 42, 37, 33, 0.0, 10, 0.0, 6, 9, 10, 10, 0.0, 0.0, 8, 9, 7, 7, 0.0, 8, 8, 9, 8, 9, 9, 10, 9, 10, 10, 10, 10, 10, 11, 11, 12, 12, 14, 0.0, 19, 22, 24, 31, 34, 39, 41, 43, 43, 41, 40, 36, 33, 29, 26, 22, 20, 18, 17, 15, 14, 13, 13, 13, 12, 12, 12, 0.0, 12, 11, 12, 12, 12, 13, 0.0, 9, 7, 0.0, 0.0, 29, 31, 35, 22, 25, 18, 29, 0.0, 36, 38, 37, 38, 38, 37, 37, 38, 37, 37, 37, 38, 37, 38, 35, 0.0, 41, 41, 37, 0.0, 34, 35, 34, 34, 33, 33, 33, 34, 32, 33, 33, 32, 34, 33, 33, 14, 0.0, 22, 22, 22, 22, 25, 26, 25, 25, 23, 23, 26, 0.0, 35, 37, 37, 37, 37, 37, 37, 37, 37, 36, 37, 37, 38, 38, 38, 38, 40, 40, 40, 39, 39, 39, 38, 38, 37, 36, 37, 36, 36, 0.0, 37, 37, 36, 36, 35, 36, 37, 38, 0.0, 0.0, 29, 30, 30, 30, 30, 31, 32, 32, 33, 34, 37, 0.0, 0.0, 39, 39, 37, 34, 0.0, 40, 24, 0.0, 29, 0.0, 17, 30, 40, 38, 26, 0.0, 25, 35, 41, 39, 38, 39, 38, 38, 38, 38, 39, 36, 0.0, 29, 29, 27, 26, 0.0, 28, 0.0, 33, 31, 30, 30, 28, 28, 27, 26, 28, 25, 24, 16, 18, 38, 41, 40])
angles = np.array([i for i in range(0, 360)])
glass_filter = GlassFilter(300)
glass_filter.coordinates = glass_filter.scan_to_points(0, 0, 0, scan)
potential_peaks = glass_filter.find_potential_peaks(intensiv=intesivities, scan=scan)
for seq in potential_peaks:
    for point in seq:
        print(f"index: {point.index} x: {point.x} y: {point.y} angle: {point.deg_angle} intensivity: {point.intensivity}")
surfaces = glass_filter.fitting(potential_peaks=potential_peaks)
for sur in surfaces:
    ps = sur.p_start
    pe = sur.p_end
    print(f"Start: {ps.x} {ps.y} End: {pe.x} {pe.y}")
new_scan, trash, occupied = glass_filter.find_real_trash(surfaces, scan, 10)
print(f"Trash: {trash}, type: {type(trash)}")
plot_scan(angles=angles, ranges=scan, intensivities=intesivities, potential_peaks=potential_peaks, threshold=300, trash=trash)
plot_topdown(glass_filter.coordinates, scan, potential_peaks)
plot_scan_comparison(scan, new_scan, occupied, surfaces)