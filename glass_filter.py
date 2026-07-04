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
    

class Window:
    def __init__(self, D1: Point, D2: Point, N: int, stdv: float = None):
        self.d1 = D1
        self.d2 = D2
        self.n = N    
        self.Uc = (D1.range - D2.range) / N
        self.stdv = stdv

class Beam:
    def __init__(self, angle: int, range: float):
        self.angle = angle
        self.range = range

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
    
    def classify_pc(self, scan: list[float], intesiv: list[float], threshold: float):
        """
         - скользящее окно размером 11 точек (5 точек до текущей точки и 5 точек после)
         - для каждого окна надо высчитать STDV дальностей
         - если STDV > `Threshold`, то точки надо пометить как кондидаты в PCoG
         - иначе пометить как кондидаты в PCoO

        Args:
            scan (list[float]): скан лидара
            intesiv (list[float]): интенсивности
            threshold (float): порог для sdtv, чтобы понять, что это PCoG или PCoO

        Returns:
            pcoo (list[Window]): массив окон объектов
            pcog (list[Window]): массив окон стекол
            stdvs (list[float]): массив стандартных отклонений
        """
        
        pcoo = []
        pcog = []
        windows = []
        stdvs = []
        for i in range(5, len(scan)):
            if i > 354:
                D1 = Point(x=self.coordinates[(i-5)%360][0], y=self.coordinates[(i-5)%360][1], index=(i-5)%360, intensivity=intesiv[(i-5)%360], range=scan[(i-5)%360], angle=[(i-5)%360])
                D2 = Point(x=self.coordinates[(i+5)%360][0], y=self.coordinates[(i+5)%360][1], index=(i+5)%360, intensivity=intesiv[(i+5)%360], range=scan[(i+5)%360], angle=[(i+5)%360])
            else:
                D1 = Point(x=self.coordinates[i-5][0], y=self.coordinates[i-5][1], intensivity=intesiv[i-5], index=i-5, range=scan[i-5], angle=i-5)
                D2 = Point(x=self.coordinates[i+5][0], y=self.coordinates[i+5][1], intensivity=intesiv[i+5], index=i+5, range=scan[i+5], angle=i+5)
            N = 11
            window = Window(D1=D1, D2=D2, N=N)
            windows.append(window)
            print(f"Start: {(i-5)%360} Center: {i} End: {(i+5)%360} Uc: {window.Uc}")
        for window in windows:
            u = 0
            for i in range(window.d1.index, window.d2.index + 1):
                u += scan[i]
            u /= window.n
            stdv = 0
            avg = 0
            for Lr in range(window.d1.index, window.d2.index + 1):
                if scan[Lr] > 0:
                    avg += np.pow(scan[Lr] - u, 2)
            avg /= window.n
            stdv = np.sqrt(avg)
            stdvs.append(stdv)
            window.stdv = stdv
            if stdv > threshold:
                pcog.append(window)
            else:
                pcoo.append(window)
        return pcoo, pcog, stdvs
    
    def check_mirrors(self, frags, scan, intensiv):
        mirrors = []
        for s, e in list(frags):
            # кусок от s-5 до e+5
            mirror_start = None
            mirror_end = None
            for i in range(s-5, s+1):
                if scan[np.abs(i%360)] - scan[np.abs((i-1)%360)] >= 225 and intensiv[np.abs(i%360)] < intensiv[np.abs((i-1)%360)]:
                    mirror_start = i
            for i in range(e, e+5):
                if scan[np.abs(i%360)] - scan[np.abs((i+1)%360)] >= 225 and intensiv[np.abs(i%360)] < intensiv[np.abs((i+1)%360)]:
                    mirror_end = i
            if mirror_end is None and mirror_start is None:
                d1 = Point(x=self.coordinates[mirror_start][0], y=self.coordinates[mirror_start][1], intensivity=intensiv[mirror_start], index=mirror_start, range=scan[mirror_start], angle=mirror_start)
                d2 = Point(x=self.coordinates[mirror_end][0], y=self.coordinates[mirror_end][1], intensivity=intensiv[mirror_end], index=mirror_end, range=scan[mirror_end], angle=mirror_end)
                mirror = Window(D1=d1, D2=d2, N=np.abs(mirror_end-mirror_start))
                mirrors.append(mirror)
        return mirrors

scan = np.array([2793.0, 1857.0, 0.0, 0.0, 1438.5, 1395.0, 1375.75, 1337.5, 1336.0, 964.0, 963.5, 967.0, 971.0, 977.0, 1005.0, 1010.5, 966.0, 930.5, 917.0, 889.0, 861.0, 849.0, 826.0, 824.5, 833.5, 842.5, 853.0, 873.5, 879.0, 890.25, 898.0, 912.25, 926.0, 926.0, 0.0, 0.0, 420.5, 393.5, 382.5, 363.0, 354.75, 341.5, 331.5, 324.5, 313.75, 310.25, 300.0, 293.25, 290.0, 286.0, 280.0, 273.5, 271.0, 269.0, 265.75, 261.5, 259.5, 256.0, 252.25, 249.75, 247.75, 247.5, 243.25, 242.0, 238.75, 238.5, 236.0, 233.0, 232.5, 232.0, 232.5, 228.5, 227.25, 227.0, 224.5, 225.5, 224.5, 223.25, 221.5, 222.5, 223.75, 222.25, 221.5, 223.0, 224.0, 225.0, 226.25, 227.0, 226.0, 227.5, 229.5, 230.25, 233.5, 234.5, 235.25, 239.0, 240.5, 241.25, 243.5, 245.0, 249.5, 251.5, 254.5, 257.0, 262.75, 265.75, 271.5, 275.0, 282.0, 286.25, 294.0, 300.5, 314.0, 330.0, 338.5, 360.0, 0.0, 265.0, 258.0, 243.5, 240.75, 208.5, 205.0, 201.5, 200.5, 199.75, 201.25, 200.5, 202.5, 205.0, 207.75, 209.75, 213.75, 216.0, 215.0, 0.0, 146.5, 0.0, 139.75, 0.0, 137.5, 133.5, 132.75, 131.5, 135.0, 137.5, 139.5, 141.0, 142.5, 148.5, 0.0, 195.25, 207.25, 210.5, 213.75, 214.0, 213.5, 212.0, 212.0, 210.75, 210.25, 212.5, 212.5, 211.0, 212.25, 215.5, 214.5, 213.75, 217.0, 218.75, 219.0, 218.5, 223.0, 218.5, 218.25, 216.5, 221.0, 222.0, 220.5, 222.5, 222.0, 220.75, 219.75, 220.0, 219.5, 218.0, 218.0, 217.5, 217.5, 219.0, 219.0, 219.5, 220.5, 218.0, 219.5, 222.5, 224.5, 227.0, 227.5, 232.0, 232.5, 237.5, 237.0, 239.5, 238.0, 237.0, 242.0, 241.0, 244.25, 243.25, 247.0, 248.75, 252.5, 254.0, 260.0, 263.0, 266.5, 275.5, 277.5, 282.0, 283.25, 284.0, 285.0, 292.5, 288.25, 0.0, 183.75, 177.0, 174.5, 168.5, 167.5, 171.5, 177.5, 179.5, 181.75, 181.25, 0.0, 0.0, 436.0, 0.0, 382.5, 376.5, 366.25, 363.75, 357.5, 356.0, 351.75, 351.75, 351.25, 351.25, 350.5, 352.0, 351.25, 353.0, 353.75, 356.0, 358.5, 361.5, 363.0, 367.0, 374.0, 376.0, 384.75, 390.5, 394.0, 401.75, 412.5, 418.5, 431.5, 437.0, 451.5, 456.5, 468.75, 476.25, 488.5, 494.0, 503.5, 511.0, 519.0, 524.0, 532.5, 536.25, 540.0, 550.0, 555.0, 561.5, 565.0, 571.0, 575.5, 580.25, 584.0, 589.0, 591.0, 599.5, 603.0, 607.5, 611.5, 612.75, 614.5, 619.75, 619.75, 620.75, 621.5, 623.25, 624.0, 619.5, 619.5, 616.0, 615.0, 610.5, 608.5, 605.5, 602.0, 598.0, 591.5, 589.0, 584.5, 582.5, 581.0, 578.5, 580.5, 580.25, 580.25, 583.5, 589.0, 592.5, 603.0, 608.5, 622.0, 634.5, 0.0, 794.5, 811.0, 838.5, 855.0, 892.0, 911.75, 954.0, 983.5, 999.5, 958.75, 958.0, 952.5, 951.5, 951.0, 0.0, 1238.75, 1266.5, 2073.5, 2071.5, 2067.25, 0.0, 974.0, 967.75, 959.0, 959.0, 961.5, 0.0, 977.0, 2782.5])
intesivities = np.array([9, 9, 0.0, 0.0, 15, 16, 15, 19, 10, 46, 46, 46, 47, 47, 50, 51, 45, 44, 44, 45, 45, 45, 46, 48, 48, 47, 46, 47, 47, 47, 46, 45, 46, 38, 0.0, 0.0, 41, 44, 44, 47, 47, 47, 48, 48, 46, 46, 46, 47, 46, 46, 45, 47, 47, 48, 48, 49, 49, 49, 50, 50, 51, 51, 52, 52, 52, 52, 52, 51, 50, 52, 50, 50, 50, 52, 52, 52, 51, 50, 52, 52, 50, 50, 50, 52, 52, 52, 51, 51, 52, 52, 51, 52, 53, 52, 52, 53, 52, 52, 51, 51, 50, 49, 48, 48, 48, 48, 48, 46, 47, 46, 46, 47, 45, 44, 44, 41, 0.0, 49, 48, 50, 49, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 51, 51, 53, 0.0, 50, 0.0, 51, 0.0, 50, 50, 51, 50, 49, 48, 48, 47, 46, 37, 0.0, 49, 51, 52, 53, 53, 53, 53, 53, 53, 53, 52, 52, 53, 52, 52, 52, 52, 52, 52, 53, 54, 54, 53, 52, 52, 51, 44, 49, 53, 54, 53, 53, 53, 52, 52, 52, 53, 52, 51, 53, 51, 53, 51, 51, 52, 51, 51, 50, 51, 50, 51, 53, 52, 51, 52, 51, 51, 51, 51, 50, 50, 50, 50, 50, 49, 48, 52, 52, 51, 51, 50, 50, 49, 45, 0.0, 51, 52, 52, 53, 52, 52, 51, 50, 43, 29, 0.0, 0.0, 43, 0.0, 48, 49, 49, 50, 51, 51, 51, 52, 52, 52, 52, 52, 51, 51, 50, 50, 49, 49, 50, 49, 51, 49, 48, 49, 48, 50, 49, 48, 47, 47, 49, 46, 46, 47, 48, 49, 47, 48, 47, 49, 47, 49, 47, 49, 49, 49, 50, 48, 50, 48, 48, 48, 47, 48, 50, 50, 48, 48, 48, 49, 48, 48, 49, 49, 49, 49, 49, 49, 48, 47, 48, 49, 49, 48, 49, 48, 48, 47, 47, 47, 48, 48, 49, 48, 48, 49, 48, 48, 48, 45, 0.0, 45, 44, 42, 42, 43, 43, 43, 50, 51, 47, 47, 46, 47, 45, 0.0, 20, 14, 15, 16, 10, 0.0, 15, 18, 22, 23, 19, 0.0, 8, 8])
angles = np.array([i for i in range(0, 360)])
glass_filter = GlassFilter(300)
glass_filter.coordinates = glass_filter.scan_to_points(0, 0, 0, scan)
pcoo, pcog, stdvs = glass_filter.classify_pc(scan, intesivities, 175)
sigma = [None]*len(scan)
for k, s in enumerate(stdvs):
    sigma[5+k] = s
frags = plot_fragments(scan, sigma, thr=175)          # уже склеенные куски
frags_after = glass_filter.check_mirros(frags, scan, intesivities)   # Фильтр 2
plot_filter2(scan, list(frags), frags_after)