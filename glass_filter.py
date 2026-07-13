import numpy as np
import matplotlib.pyplot as plt
from algorithms import *


class Beam:
    def __init__(self, range, index, intensiv):
        self.range = range
        self.index = index
        self.intensiv = intensiv

class Point:
    def __init__(self, x, y, index, range = None, intensiv = None):
        self.x = x
        self.y = y
        self.index = index
        self.range = range
        self.beam = Beam(self.range, index, intensiv)

class Window:
    def __init__(self, s, e, cb, stdv = None, array = None):
        self.s = s
        self.e = e
        self.stdv = stdv
        self.cb = cb
        self.array = array

class PointCloud:
    def __init__(self, points: list[Point]):
        self.points = points

class GlassFilter:
    def __init__(self, scan_len: int = 360):
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._len = scan_len
    
    def set_len(self, nlen: int = 180):
        """Поменять на лету длину скана, который надо обработать

        Args:
            nlen (int, optional): Новая длина скана. Defaults to 180.
        """
        self._len = nlen
        
    def set_xyt(self, nx, ny, ntheta):
        """Поменять текущие координаты робота для расчетов.

        Args:
            nx (float): новый x.
            ny (_type_): новый y.
            ntheta (_type_): новый theta.
        """
        self._x = nx
        self._y = ny
        self._theta = ntheta
    
    def scan_to_points(self, scan: list[int]) -> list[list[int]]:
        """Функция для расчета координат точек.

        Args:
            scan (list[int]): скан.

        Returns:
            list: список со списками координат [[x1, y1],[x2, y2]]
        """
        coordinates = []  # [[x, y], [x1, y1]]
        for i in range(len(scan)):
            xo = self._x + scan[i] * np.cos(self._theta + np.deg2rad(i))
            yo = self._y + scan[i] * np.sin(self._theta + np.deg2rad(i))
            coordinates.append([xo, yo])
        return coordinates
    
    def group_candidates(self, mbeam_indices, glue_gap=4):
        """склеить сырые кандидаты в одно целое, если получается.

        Args:
            mbeam_indices (list[Beam]): список лучей
            glue_gap (int, optional): Максимальный разрыв между лучами. Defaults to 4.

        Returns:
            list[list[int]]: промежутки [[s1, e1], [s2, e2]]
        """
        if not mbeam_indices:
            return []
        segments = []
        s = e = mbeam_indices[0].index
        for beam in mbeam_indices[1:]:
            ind = beam.index
            gap = (ind - e) % self._len
            if gap <= glue_gap:
                e = ind
            else:
                segments.append([s, e])
                s = e = ind
        segments.append([s, e])
        if len(segments) > 1:
            fs, fe = segments[0]
            ls, le = segments[-1]
            if (fs - le) % self._len <= glue_gap:
                segments[0] = [ls, fe]
                segments.pop()
        return segments
    
    def stdv_filter(self, scan: list[float], intensiv: list[float], threshold: float):
        """Алгоритм 1: фильтр скана по скользящим окнам и STDV (СКО)
        Взято из статьи на MDPI

        Args:
            scan (list[float]): скан лидара.
            intensiv (intensiv[float]): интенсивность скана.
            threshold (float): порог STDV после которого объект считается "стеклом".

        Returns:
            pcs (list[PointCloud]): облака точек, которые содержат точки "стекла".
            invalids (list[int]): инвалидные индексы.
            clued_candidates (list[list[int]]): промежутки, которые засчитались за стекла.
        """
        # концепция скользящего окна по скану от 5 элемента до len(конец)+5
        coords = self.scan_to_points(scan)
        mbeams = []
        invalids = []
        for i in range(len(scan)):
            if scan[i] <= 0:
                invalids.append(i)
        
        # 5 точек до текущей точки, текущая точка и 5 точек после точки
        
        for i in range(5, len(scan)+5):
            j = i%self._len # я постоянно путался что скан то разрывается поле 359 элемента
            count = 11 # пока что в окне все лучи нормальные
            r = scan[j]
            I = intensiv[j]
            # концепция СКО (STDV). Ее надо посчитать по все окну
            # [359, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10] - это пример окна когда надо постороить правильный расчет
            # если центральный луч с индексом < 5 или брльше 354, то надо выкручивать j до того чтобы он стал в переделах 0..359
            # j=0, край j-5=-5, -5%self._len=-5-self._len*(-5//self._len)=355
            start = (j-5)%self._len
            end = (j+5)%self._len
            # газ считать STDV
            # тот массив свхеру с индексами, нужно нормализовать
            dev = self._len-start
            inds = [(k-dev)%self._len for k in range((start+dev)%self._len, (end+dev)%self._len)]
            inds.sort()
            for ind in inds.copy():
                if ind in invalids:
                    count-=1 # луч портит ско
                    inds.remove(ind)
            avg = sum([scan[k] for k in inds]) / count # сумма дальностей
            stdv = np.sqrt(sum([(scan[k]-avg)**2 for k in inds]) / count)
            if stdv > threshold: # если отклонение больше нормы, то окно надо пометить
                mbeam = Beam(r, j, I)
                mbeams.append(mbeam)
        
        # склейка лучей в обрывки
        # нашелся еще один баг: склека склеивает только последовательные окна, а те которые черезз 2 и более индексов пропускает
        candidates = []
        s = e = mbeams[0].index # я не заметил, как 2 недели жил с критическим багом (s=e=0)
        prev_ind = mbeams[0].index
        pcs = []
        for i in range(1, len(mbeams)):
            ind = mbeams[i].index
            #print(ind)
            if ind - prev_ind == 1:
                e = ind
            else:
                candidates.append([s, e])
                s = ind
                e = ind
            prev_ind = ind
        # доклейка кусков окон через 2-3 индекс в одно целое

        clued_candidates = self.group_candidates(mbeams, 4)
        for can in clued_candidates:
            s, e = can
            steps = (e - s) % self._len + 1
            points = []
            for j in range(steps+1):
                i = (s + j) % self._len
                x = coords[i][0]
                y = coords[i][1]
                p = Point(x, y, i, scan[i], intensiv[i])
                points.append(p)
            pc = PointCloud(points)
            pcs.append(pc)
        clued_candidates.sort(reverse=True)
        print(f"new candidates: {clued_candidates}")
        return pcs, invalids, clued_candidates
    
    def find_frame(self, scan: list[float], start_idx: int, direction: int = 1, max_steps: int = 20):
        """Найти раму окна со стартовой позиции start_idx в направлении direction, пройдя максимум max_steps шагов.

        Args:
            scan (list[float]): скан лидара.
            start_idx (int): стартовый индес для поиска рамы.
            direction (int, optional): Направление, в котором искать раму (1 или -1). Defaults to 1.
            max_steps (int, optional): Максимальное количество шагов в направлении direction, чтобы найти раму. Defaults to 20.

        Returns:
            list[]: Значения рамы в scan и ее индекс в scan, либо [None, None], если не нашлась
        """
        ind = start_idx
        for _ in range(max_steps):
            ind = (ind + direction) % len(scan)
            if scan[ind] > 0:
                return scan[ind], ind
        return None, None   # рама не нашлась за разумное число шагов
    
    def valid_filter(self, pc: PointCloud, min_points: int = 7, min_amp: float = 2.0, max_chng: float = 100.0, kszie: int = 3, kmediansize: int = 25, rdp: float = 0.3):
        """Фильтр по физическим свойствам стекла: длина, амплитуда скачка дальностей, максимальное отклонение дальностей, непрерывность стекла, интенсивность сначала падает, а затем растет обратно.
        (На 2д лидаре ловит даже не стекло)
        
        Args:
            pc (PointCloud): облако точек для проверки на ваидность.
            min_points (int, optinal): минимальная длина стекла. Defaults to 7.
            min_amp (float, optinal): минимальная амплитуда скачка в стекле. Defaults to 2.0.
            max_chng (float, optinal): максимальный разрыв между точками в стекле (проверка на непрерывность). Defaults to 100.0.
            ksize (int, optinal): ядро для сглаживания отрывка интенсивностей в окне (numpy convolve). Defaults to 3.
            kmediansize (int, optinal): ядро для сглаживания отрывка интенсвинсотей в окне (медианный фильтр). Defaults to 25.
            rdp (float, optional): доля (процента / 100) амплетуды, чтобы засчитать падение и подъем интенсивности. Defaults to 0.3.
        """
        
        
        """
        форма пика: интенсивность точек в sequence  должна сначала монотонно возрастать, а затем убывать
        минимальная ширина: т.к. пик интенсивности это практически всегда мусор, то длина последовательности должна быть маленькой
        амплитуда: разница между максимальной интенсивностью и минимальной должна быть больше заданного порога
        физическая непрерывность: расстояние между лидаром и точками sequene не должна резко меняться.
        """
        # фильтруем сначала по ширине последовательности, потому что дальше есть функции, которые работают с итераторами в списках. МОжет быть IndexErro или какой-нибудь дргуой краш
        #print(f"Последовательность: {pc.points[0].index} {pc.points[-1].index}")
        # ширина
        if len(pc.points) < min_points:
            print(f"len is invalid: {len(pc.points)}")
            return False, None

        # новый критерий: дропауты интенсивности
        intesiv_scrap = [pc.points[i].beam.intensiv for i in range(len(pc.points))]
        # if not 0 in intesiv_scrap:
        #     #print(f"no dropouts")
        #     return False, None

        # амплитуда:
        # медианный фильтр
        median = median_filter(intesiv_scrap, kmediansize)
        dno_idx = np.argmin(np.where(median == 0, np.inf, median))
        smoothed = np.convolve(median, np.ones(kszie)/kszie, mode='same') # сглаживание
        # дно
        dno = smoothed[dno_idx]
        global_dno_idx = (pc.points[0].index + dno_idx)%self._len 
        n = len(intesiv_scrap)
        edge_size = max(2, int(n * 0.2)) # специальная защита: нельзя брать размер краев за 1 индекс, минимум 2
        # нахождение диапозонов scrap'а, где надо найти максимумы
        
        sind = np.argmax(intesiv_scrap[:edge_size])
        start_max = intesiv_scrap[sind]

        local_send = np.argmax(intesiv_scrap[-edge_size:])
        # Перевод во всеобщий (глобальный) индекс исходного массива
        send = (n - edge_size) + local_send
        end_max = intesiv_scrap[send]

        amp = max(start_max, end_max) - dno
        if amp < min_amp:
            print(f"amp is invalid: {amp}")
            return False, None

        # непрерывность
        d = prevd = dd = 0
        dds = []
        cont_valid = True
        for i in range(0, len(pc.points)):
            point = pc.points[i]
            d = np.sqrt((point.x - self._x)**2 + (point.y - self._y)**2)
            if i == 0:
                prevd = d
                continue
            dd = prevd - d
            dds.append(dd)
            if dd > max_chng:
                cont_valid = False
            prevd = d
        #print(f"dds: {dds}")
        if not cont_valid:
            #print(f"contour is invalid: {dd}")
            return False, None

        # возрастание и убывание
        """
        На изображении intensivities_correlation.png видно, что стекло на самом деле находится в промежутке 357-8.
        Можно увидеть, что образовался перевернутый конус из интенсивностей, до момента, когда, дальности не оказались на стекле. 
        Но растет и падает она с переменным успехом, поэтому надо научиться детектить такой подъем
        """
        
        dropped = (start_max - dno) > (amp * 0.3)
        recovered = (end_max - dno) > (amp * 0.3)
        # аоследовательность упала и выросла обратно (shape valid)
        if dropped and recovered:
            print(f"последовательность упала и выросла обратно, старт: {pc.points[0].index}, дно: {global_dno_idx}, конец: {pc.points[-1].index})")
            return True, global_dno_idx
        return False, None
    
    def fitting_filter(self, scan: list[int], inds: list[int], min_var_ratio: float = 0.8):
        """Фильтр-фитинг прямой через оставшихся кондидатов на звание PCOG при помощи алгоритма главных компонент PCA

        Args:
            scan (list[int]): скан лидара.
            inds (list[int]): три индекса: левая рама, середина, правая рама
            min_var_ratio (float, optional): _description_. Defaults to 0.87.

        Returns:
            _type_: _description_
        """
        coordinates = self.scan_to_points(scan)
        coords = []
        coords.append([coordinates[inds[0]][0], coordinates[inds[0]][1]])
        coords.append([coordinates[inds[1]][0], coordinates[inds[1]][1]])
        coords.append([coordinates[inds[2]][0], coordinates[inds[2]][1]])
        x_mean, k, b, var_ratio = pca(coords, num_components=2) # фит по двум главным компонентам
        print(f"x mean: {x_mean} k: {k} b: {b}")
        # var_ratio - это отношение дисперсий, от 0 до 1, Показывает, как много данных получается зафитить (грубо говоря).
        # Если доля или процент высокий, то это значит, что почти все точки идеально ложатся на прмую, если низкий, то значит, что это хаотично разбросанные точки - мусор.
        if var_ratio >= min_var_ratio:
            print(f"Normal variance ratio: {var_ratio}")
        else:
            print(f"Bad variance ratio: {var_ratio}")
        
        return x_mean, k, b, coords, var_ratio
        

    def accum_scans(self, scans: list[list[int]]):
        """Функция для накопления сканов. Желательно накапливать от 3 и более сканов.
        Работает по принципу голосования за каждый элемент скана.

        Args:
            scans (list[list[int]]): список сканов. Желательно от 3.

        Returns:
            _type_: _description_
        """
        voices = [0]*len(scans[0])
        truth = [0]*len(scans[0])
        for sc in scans:
            pcs, _ = self.stdv_filter(sc, 175, self._x, self._y, self._theta)
            for pc in pcs:
                for p in pc.points:
                    voices[p.index] += 1
        # ответ: где стоит стекло
        for i in range(len(scans[0])):
            truth[i] = voices[i] >= np.floor(len(scans) * 0.5)
            # стекло там где количество подтвержденных ответов стекло тут или нет превысило 50% от количества сканов
        return truth 
    
    def patch(self, pcog: list[PointCloud], scan: list[int], dif_ratio: float = 0.3):
        """Функция превращения стекла в стену, замазка куска скана.

        Args:
            pcog (list[PointCloud]): Список облаков точек.
            scan (list[int]): Скан лидара, который надо замазать. 
            dif_ratio (float, optional): Максимальная разница между двумя рамами в долях. Defaults to 0.3.

        Returns:
            _type_: _description_
        """
        # нашелся один баг: замазка работает плохо, поэтому надо взять раму и замазать ей не только коно, но еще и по одной точке перед и после него
        new_scan = scan.copy()
        occupied = [0]*len(new_scan)
        for pc in pcog:
            s, e = pc.points[0].index, pc.points[-1].index
            # ищем подходящие друг другу рамы по новому алгоритму поиска
            win_frame_left, il = self.find_frame(scan, s, -1, 12)
            win_frame_right, ir = self.find_frame(scan, e, 1, 12)
            print(f"Frames: left {win_frame_left} at {il} and right {win_frame_right} at {ir}")
            if il is None or ir is None:
                continue
            dif = abs(win_frame_left - win_frame_right) / max(win_frame_right, win_frame_left) # смотрим какую часть составляет одна рама от другой
            print(f"Difference between frame: {dif}")
            if dif < dif_ratio: # если рамы похожи
                frame = (win_frame_left + win_frame_right) / 2 # так работает намного лучше
            else: 
                continue # пропустить замазку
            steps = (e - s) % self._len + 1
            #print(f"lframe: {win_frame_left} r_frame: {win_frame_right} s: {s} e: {e} points count: {steps}")
            
            for step in range(steps):
                p = (s + step) % self._len
                new_scan[p] = frame
                occupied[p] = 1
        return new_scan, occupied

scan = np.array([807.0, 822.0, 821.5, 821.5, 817.0, 774.0, 778.0, 779.5, 783.25, 769.5, 765.5, 774.0, 0.0, 0.0, 639.25, 645.5, 653.5, 658.0, 654.5, 658.5, 662.0, 668.5, 672.0, 682.5, 684.75, 687.0, 695.5, 702.5, 705.25, 722.0, 722.0, 725.75, 735.25, 741.5, 756.75, 765.5, 771.25, 786.0, 793.0, 812.0, 822.5, 829.5, 848.5, 858.75, 851.75, 893.25, 903.5, 926.5, 932.5, 929.0, 918.0, 927.0, 905.0, 890.5, 875.75, 860.0, 854.0, 839.5, 826.25, 821.0, 810.0, 804.0, 794.0, 785.5, 780.5, 772.25, 770.0, 761.5, 781.0, 800.0, 837.0, 855.0, 900.5, 926.5, 0.0, 0.0, 1096.75, 1089.0, 1077.75, 1076.25, 1073.5, 1072.0, 1068.0, 1067.0, 1065.5, 1066.0, 1064.25, 1063.25, 1064.0, 1062.5, 1062.0, 1065.5, 1066.25, 1065.25, 1058.25, 1053.75, 1058.25, 1065.25, 1079.5, 1084.5, 1087.5, 1092.75, 1094.75, 1100.5, 1103.0, 1109.0, 1117.25, 1120.0, 1147.5, 1173.5, 1209.5, 1160.5, 1155.0, 1160.0, 1125.0, 1064.0, 0.0, 1234.5, 1248.5, 1256.75, 1273.5, 1283.25, 1302.0, 1310.75, 1332.5, 1340.0, 1354.0, 1380.5, 1357.0, 1351.5, 0.0, 1412.25, 1432.5, 1447.75, 1433.75, 1430.25, 1485.0, 0.0, 1376.5, 1354.0, 1340.0, 1310.5, 1294.0, 1292.5, 1283.0, 1279.5, 1271.0, 1260.25, 1256.0, 1259.75, 0.0, 1217.5, 1205.0, 1221.0, 1305.25, 1299.25, 1289.0, 1288.5, 1280.0, 1271.5, 1260.5, 1090.0, 1086.5, 1085.75, 1097.5, 1102.5, 1241.5, 1249.0, 1251.0, 1249.5, 1245.75, 1249.5, 1249.0, 1043.25, 1031.5, 1021.25, 1018.25, 1024.0, 1038.5, 1254.0, 1258.25, 1260.0, 0.0, 625.0, 1257.0, 1277.5, 1284.5, 1294.5, 1295.5, 1301.25, 1314.0, 1316.0, 1321.5, 1335.5, 1345.5, 1352.0, 1368.25, 1374.5, 804.75, 0.0, 1406.0, 1423.75, 1433.0, 1455.25, 1463.5, 1490.0, 1497.5, 0.0, 559.5, 568.5, 1304.5, 1283.0, 1246.0, 1226.5, 1194.0, 1176.75, 1144.0, 1116.75, 1104.0, 1073.5, 1059.0, 1046.5, 992.25, 955.5, 942.5, 930.5, 914.0, 901.5, 885.0, 877.75, 865.5, 855.5, 842.0, 828.5, 825.5, 806.75, 800.5, 791.5, 789.0, 780.5, 771.5, 763.0, 755.0, 752.5, 751.0, 742.75, 735.75, 731.0, 723.75, 721.0, 722.5, 720.75, 707.5, 704.75, 702.0, 695.5, 694.5, 693.25, 689.75, 686.0, 685.5, 684.0, 682.25, 687.0, 692.25, 680.75, 681.25, 685.0, 683.0, 681.0, 681.5, 689.5, 686.0, 680.0, 682.0, 682.5, 683.0, 685.0, 691.5, 688.5, 692.0, 693.0, 699.5, 701.0, 711.25, 713.25, 710.0, 711.75, 718.0, 720.25, 725.0, 733.5, 733.5, 739.5, 750.0, 753.0, 763.5, 770.5, 772.5, 778.5, 786.5, 793.0, 811.0, 812.0, 822.25, 829.0, 818.5, 850.5, 864.75, 872.5, 0.0, 839.5, 840.5, 847.25, 853.75, 854.0, 839.5, 824.5, 786.5, 792.5, 786.5, 763.0, 757.0, 744.5, 741.5, 741.5, 746.5, 771.0, 785.0, 815.0, 832.0, 858.0, 859.5, 863.5, 851.75, 849.0, 0.0, 894.25, 3253.5, 3457.0, 0.0, 0.0, 0.0, 3717.25, 0.0, 3876.75, 0.0, 1974.5, 1976.0, 2003.5, 2003.0, 1994.5, 1982.0, 1991.0, 1993.5, 1985.0, 1980.75, 0.0, 805.0, 798.0])
intensivities = np.array([34, 19, 14, 18, 22, 46, 47, 46, 49, 47, 48, 52, 0.0, 0.0, 48, 50, 50, 50, 49, 49, 48, 49, 49, 49, 47, 47, 46, 46, 45, 49, 46, 45, 45, 46, 47, 47, 45, 45, 45, 46, 43, 44, 45, 46, 46, 44, 45, 47, 45, 44, 41, 43, 39, 38, 39, 39, 39, 39, 40, 41, 41, 41, 41, 41, 42, 43, 42, 41, 32, 28, 27, 27, 28, 25, 0.0, 0.0, 50, 46, 45, 46, 46, 46, 47, 47, 47, 48, 47, 48, 48, 48, 48, 48, 48, 46, 45, 44, 45, 45, 47, 47, 47, 47, 46, 46, 45, 46, 45, 46, 46, 44, 46, 47, 48, 49, 24, 22, 0.0, 47, 47, 45, 45, 46, 45, 45, 46, 47, 45, 44, 33, 30, 0.0, 44, 40, 38, 34, 33, 34, 0.0, 44, 45, 44, 44, 46, 45, 46, 46, 47, 45, 45, 41, 0.0, 40, 45, 37, 48, 48, 47, 47, 47, 46, 41, 43, 45, 45, 45, 40, 48, 48, 49, 48, 49, 49, 48, 40, 44, 46, 45, 43, 31, 48, 48, 47, 0.0, 47, 43, 47, 47, 47, 46, 47, 47, 46, 46, 45, 46, 47, 46, 47, 47, 0.0, 45, 43, 44, 45, 44, 47, 47, 0.0, 47, 31, 39, 39, 41, 40, 41, 42, 43, 44, 44, 44, 43, 44, 43, 46, 46, 46, 45, 45, 45, 47, 46, 45, 45, 47, 48, 47, 46, 47, 47, 47, 47, 48, 48, 47, 49, 48, 48, 48, 48, 48, 50, 50, 49, 48, 48, 47, 48, 49, 49, 49, 49, 49, 49, 50, 50, 49, 49, 50, 50, 50, 49, 51, 49, 49, 48, 48, 48, 49, 50, 49, 50, 50, 50, 50, 51, 49, 50, 49, 49, 49, 48, 51, 47, 48, 48, 49, 49, 47, 47, 47, 47, 46, 48, 47, 47, 47, 45, 46, 47, 46, 0.0, 46, 46, 46, 45, 46, 44, 44, 37, 45, 45, 43, 44, 44, 46, 44, 44, 43, 43, 44, 44, 47, 47, 47, 45, 43, 0.0, 20, 9, 9, 0.0, 0.0, 0.0, 6, 0.0, 6, 0.0, 26, 23, 26, 26, 24, 23, 22, 23, 17, 14, 0.0, 34, 37])    
SCAN_LEN = len(scan)
gf = GlassFilter(SCAN_LEN)
gf.set_xyt(0.0, 0.0, 0.0)
pcs, invalids, candidates = gf.stdv_filter(scan, intensivities, 175)
angles = [a for a in range(0, SCAN_LEN)]
founded_inds = []
window = 15
pca_data = []

print(f"scan: {scan}")
print(f"ints: {intensivities}")

for pc in pcs.copy():
    valid, dno_idx = gf.valid_filter(pc=pc, max_chng=10000.0, kmediansize=window, min_points=3, min_amp=10)
    if not valid:
        pcs.remove(pc)
        continue
    start = pc.points[0].index
    end = pc.points[-1].index
    _, k, b, coords, var_ratio = gf.fitting_filter(scan, [start, dno_idx, end])
    print(f"Fitting: {k} {b} {coords} {var_ratio}")
    if var_ratio < 0.8:
        pcs.remove(pc)
        continue
    for p in pc.points:
        pca_data.append([k, b, coords, var_ratio, [start, dno_idx, end]])
        founded_inds.append(p.index)

sc, oc = gf.patch(pcs, scan)
smoothed = np.convolve(intensivities, np.ones(window)/window, mode='same')
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.scatter(np.radians(angles), sc, color='blue', s=15, label='замеры')
# ax.scatter(np.radians(angles), smoothed * 20, color="purple", s=5, label="интенсивности")
ax.scatter(0, 0, color='red', s=120, marker='*', label='лидар')
founded_angles = np.radians(np.array(angles)[founded_inds])
founded_dists = np.array(sc)[founded_inds]

label = False
for data in pca_data:
    k, b, coords, var_ratio, inds = data
    seg_points = None
    for item in data:
        if isinstance(item, (list, np.ndarray)) and len(np.shape(item)) == 2:
            seg_points = np.asarray(item)
            break
            
    if seg_points is None:
        seg_angles = sc[inds]
        seg_distances = sc[inds]
        seg_points = np.column_stack((seg_distances * np.cos(seg_angles), seg_distances * np.sin(seg_angles)))

    mean_x = np.mean(seg_points[:, 0])
    mean_y = np.mean(seg_points[:, 1])
    k = data[0] if isinstance(data[0], (int, float)) and not isinstance(data[0], bool) else data[1]

    if np.isinf(k) or k == float('inf') or np.isnan(k):
        y_line = np.linspace(min(seg_points[:, 1]) - 0.5, max(seg_points[:, 1]) + 0.5, 100)
        x_line = np.full_like(y_line, mean_x) 
    else:
        x_line = np.linspace(min(seg_points[:, 0]) - 0.5, max(seg_points[:, 0]) + 0.5, 100)
        b_safe = mean_y - k * mean_x
        y_line = k * x_line + b_safe

    to_polar = lambda x, y: (np.arctan2(y, x), np.sqrt(x**2 + y**2))
    theta_pts, r_pts = to_polar(seg_points[:, 0], seg_points[:, 1])
    theta_mean, r_mean = to_polar(mean_x, mean_y)
    theta_line, r_line = to_polar(x_line, y_line)

    # if not label:
    #     ax.scatter(theta_pts, r_pts, color='red', s=25, zorder=5, label="точки для фитинга")
    #     ax.plot(theta_line, r_line, color='black', linestyle='--', linewidth=2, label="линия фита")
    #     label = True
    # else: 
    #     ax.scatter(theta_pts, r_pts, color='red', s=25, zorder=5)
    #     ax.plot(theta_line, r_line, color='black', linestyle='--', linewidth=2)
        

ax.scatter(invalids, [100] * len(invalids), color="red", s=1, marker="o", label='дропауты, выдвеннутые вперед')
ax.scatter(founded_angles, founded_dists, color="green", s=12, marker="o", label='стекла')
plt.title("карта комнаты", pad=20, fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()