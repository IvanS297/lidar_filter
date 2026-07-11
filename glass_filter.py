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
    def __init__(self):
        pass
    
    def scan_to_points(self, x: float, y: float, theta: float, scan: list) -> list:
        coordinates = []  # [[x, y], [x1, y1]]
        for i in range(len(scan)):
            xo = x + scan[i] * np.cos(theta + np.deg2rad(i))
            yo = y + scan[i] * np.sin(theta + np.deg2rad(i))
            coordinates.append([xo, yo])
        return coordinates
    
    def stdv_filter(self, scan: list, intensiv: list, threshold: float, x: float, y: float, theta: float):
        """Алгоритм 1: фильтр скана по скользящим окнам и STDV (СКО)

        Args:
            scan (list): скан лидара
            intensiv (intensiv): интенсивность скана
            threshold (float): порог STDV после которого объект считается "стеклом"
            x (float): x робота
            y (float): y робота
            theta (float): theta робота

        Returns:
            PCoG (list): облака точек, которые содержат точки "стекла"
        """
        # концепция скользящего окна по скану от 5 элемента до len(конец)+5
        coords = self.scan_to_points(x, y, theta, scan)
        mbeams = []
        invalids = []
        for i in range(len(scan)):
            if scan[i] <= 0:
                invalids.append(i)
        
        # 5 точек до текущей точки, текущая точка и 5 точек после точки
        
        for i in range(5, len(scan)+5):
            j = i%360 # я постоянно путался что скан то разрывается поле 359 элемента
            count = 11 # пока что в окне все лучи нормальные
            r = scan[j]
            I = intensiv[j]
            # концепция СКО (STDV). Ее надо посчитать по все окну
            # [359, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10] - это пример окна когда надо постороить правильный расчет
            # если центральный луч с индексом < 5 или брльше 354, то надо выкручивать j до того чтобы он стал в переделах 0..359
            # j=0, край j-5=-5, -5%360=-5-360*(-5//360)=355
            start = (j-5)%360
            end = (j+5)%360
            # газ считать STDV
            # тот массив свхеру с индексами, нужно нормализовать
            dev = 360-start
            inds = [(k-dev)%360 for k in range((start+dev)%360, (end+dev)%360)]
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
        s = e = mbeams[0].index
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
        clued_candidates = []
        used_inds = []
        for can in range(len(candidates)):
            #print(candidates[can])
            ps, pe = candidates[(can-1)%len(candidates)]
            s, e = candidates[can]
            #print(f"ps: {ps} pe: {pe} s: {s} e: {e}")
            #print(f"(s-pe)%360: {(s-pe)%360}")
            if (s-pe)%360 < 3:
                #print(f"new: {ps} {e}")
                clued_candidates.append([ps, e])
                used_inds.append(s)
                used_inds.append(e)
                points = []
                steps = (e - ps) % 360 + 1

                for j in range(steps+1):
                    i = (ps + j) % 360
                    x = coords[i][0]
                    y = coords[i][1]
                    p = Point(x, y, i, scan[i], intensiv[i])
                    points.append(p)
                pc = PointCloud(points)
                pcs.append(pc)
            else:
                if s in used_inds or e in used_inds or ps in used_inds or pe in used_inds:
                    continue
                clued_candidates.append([s, e])
                points = []
                for j in range(s, e+1):
                    x = coords[j][0]
                    y = coords[j][1]
                    p = Point(x, y, j, scan[j], intensiv[j])
                    points.append(p)
                pc = PointCloud(points)
                pcs.append(pc)

        clued_candidates.sort(reverse=True)
        print(f"new candidates: {clued_candidates}")
        return pcs, invalids, clued_candidates
    
    def find_frame(self, scan, start_idx, direction, max_steps=20):
        ind = start_idx
        for _ in range(max_steps):
            ind = (ind + direction) % len(scan)
            if scan[ind] > 0:
                return scan[ind], ind
        return None, None   # рама не нашлась за разумное число шагов
    
    def valid_filter(self, min_points: int, min_amp: float, x: float, y: float, max_chng: float, pc: PointCloud, window: int):
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
            #print(f"len is invalid: {len(pc.points)}")
            return False, None

        # новый критерий: дропауты интенсивности
        intesiv_scrap = [pc.points[i].beam.intensiv for i in range(len(pc.points))]
        # if not 0 in intesiv_scrap:
        #     #print(f"no dropouts")
        #     return False, None

        # амплитуда:
        # медианный фильтр
        median = median_filter(signal=intesiv_scrap, kernel_size=3)
        dno_idx = np.argmin(np.where(median == 0, np.inf, median))
        smoothed = np.convolve(median, np.ones(window)/window, mode='same') # сглаживание
        # дно
        dno = smoothed[dno_idx]
        global_dno_idx = (pc.points[0].index + dno_idx)%360 
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
            #print(f"amp is invalid: {amp}")
            return False, None

        # непрерывность
        d = prevd = dd = 0
        dds = []
        cont_valid = True
        for i in range(0, len(pc.points)):
            point = pc.points[i]
            d = np.sqrt((point.x - x)**2 + (point.y - y)**2)
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
    
    def fitting_filter(self, scan: list[float], inds: list[int]):
        coordinates = self.scan_to_points(0, 0, 0, scan)
        coords = []
        coords.append([coordinates[inds[0]][0], coordinates[inds[0]][1]])
        coords.append([coordinates[inds[1]][0], coordinates[inds[1]][1]])
        coords.append([coordinates[inds[2]][0], coordinates[inds[2]][1]])
        x_mean, k, b, var_ratio = pca(coords, num_components=2)
        print(f"x mean: {x_mean} k: {k} b: {b}")
        if var_ratio >= 0.87:
            print(f"Normal variance ratio: {var_ratio}")
        else:
            print(f"Bad variance ratio: {var_ratio}")
        
        return x_mean, k, b, coords, var_ratio
        

    def accum_scans(self, scans, x: float, y: float, theta: float):
        voices = [0]*len(scans[0])
        truth = [0]*len(scans[0])
        for sc in scans:
            pcs, _ = self.stdv_filter(sc, 175, x, y, theta)
            for pc in pcs:
                for p in pc.points:
                    voices[p.index] += 1
        # ответ: где стоит стекло
        for i in range(len(scans[0])):
            truth[i] = voices[i] >= np.floor(len(scans) * 0.5)
            # стекло там где количество подтвержденных ответов стекло тут или нет превысило 50% от количества сканов
        return truth 
    
    def patch(self, pcog, scan):
        # нашелся один баг: замазка работает плохо, поэтому надо взять раму и замазать ей не только коно, но еще и по одной точке перед и после него
        new_scan = scan.copy()
        occupied = [0]*len(new_scan)
        for pc in pcog:
            s, e = pc.points[0].index, pc.points[-1].index
            win_frame_left, il = self.find_frame(scan, s, -1, 12)
            win_frame_right, ir = self.find_frame(scan, e, 1, 12)
            if il is None or ir is None:
                continue
            dif = abs(win_frame_left - win_frame_right) / max(win_frame_right, win_frame_left)
            if dif < 0.3:
                frame = (win_frame_left + win_frame_right) / 2 # так работает намного лучше
            else: 
                continue
            steps = (e - s) % 360 + 1
            #print(f"lframe: {win_frame_left} r_frame: {win_frame_right} s: {s} e: {e} points count: {steps}")
            
            for step in range(steps):
                p = (s + step) % 360
                new_scan[p] = frame
                occupied[p] = 1
        return new_scan, occupied

scan = np.array([278.75, 283.75, 289.5, 296.5, 303.0, 315.5, 325.75, 328.5, 332.0, 332.0, 0.0, 0.0, 209.0, 208.5, 207.0, 206.0, 205.0, 202.75, 203.0, 203.5, 203.0, 205.5, 206.0, 207.5, 211.5, 215.0, 221.0, 229.0, 0.0, 240.0, 245.75, 255.5, 257.5, 265.75, 274.0, 276.5, 280.5, 282.25, 283.5, 284.5, 286.5, 284.0, 285.0, 282.25, 281.5, 279.75, 279.75, 276.75, 276.0, 267.0, 265.0, 261.0, 0.0, 0.0, 199.5, 197.5, 194.5, 192.5, 188.5, 185.75, 185.75, 183.75, 182.0, 184.0, 182.5, 183.0, 183.5, 182.5, 183.5, 188.0, 190.75, 196.25, 201.5, 205.0, 207.5, 209.5, 215.0, 214.5, 216.5, 217.5, 221.5, 225.0, 230.5, 233.25, 259.0, 260.5, 0.0, 210.5, 210.0, 214.75, 218.0, 221.75, 227.0, 0.0, 374.5, 381.75, 395.0, 399.5, 406.5, 410.75, 419.25, 421.25, 425.75, 426.75, 423.25, 420.0, 417.5, 1298.5, 1296.0, 1287.0, 1282.5, 1281.5, 0.0, 1195.5, 1183.5, 1158.0, 0.0, 1051.5, 1043.0, 1019.0, 1047.25, 1094.0, 1108.5, 1089.0, 1082.5, 1075.25, 1070.5, 1063.0, 1059.5, 1052.25, 1046.75, 1043.0, 1040.5, 1051.5, 0.0, 1172.25, 1175.5, 1178.5, 1177.5, 1025.0, 1017.5, 1017.5, 1018.0, 1022.0, 1028.0, 1028.5, 1014.0, 994.0, 968.0, 966.5, 968.5, 1100.0, 1100.75, 1116.25, 1134.5, 1316.0, 1318.25, 1308.5, 0.0, 1660.0, 1660.5, 1664.0, 1667.0, 1672.75, 1676.75, 1683.0, 1689.0, 1696.75, 1701.0, 1710.0, 1721.25, 1725.75, 1739.0, 1745.75, 1757.25, 1767.0, 1782.0, 1790.0, 1805.5, 1814.0, 1832.5, 1844.5, 1868.25, 1879.5, 1889.0, 1446.5, 1441.25, 1447.25, 1008.0, 981.75, 967.75, 954.0, 942.25, 914.5, 882.5, 873.5, 853.5, 842.0, 819.5, 810.5, 791.5, 783.5, 767.5, 759.5, 746.5, 740.25, 725.5, 718.0, 707.5, 697.5, 691.0, 679.25, 674.75, 664.25, 657.0, 651.25, 646.0, 639.0, 638.0, 633.75, 631.25, 624.0, 621.5, 613.25, 611.0, 606.0, 603.5, 600.5, 597.75, 596.5, 594.0, 589.0, 586.0, 587.0, 585.0, 581.25, 582.5, 584.0, 583.0, 580.0, 578.75, 576.75, 576.5, 580.0, 580.0, 580.5, 578.0, 579.5, 581.5, 582.0, 582.0, 584.5, 584.75, 582.75, 584.75, 586.5, 587.0, 588.5, 592.5, 597.0, 597.75, 602.75, 603.0, 604.5, 612.5, 614.5, 618.0, 622.5, 628.25, 632.0, 636.5, 640.5, 648.5, 653.25, 661.0, 665.75, 676.0, 680.5, 691.0, 695.25, 708.75, 714.75, 721.0, 732.75, 740.0, 754.0, 762.5, 776.5, 783.5, 804.0, 811.5, 828.0, 840.5, 863.5, 875.0, 898.75, 910.75, 944.0, 955.5, 987.0, 1003.0, 1019.5, 1058.0, 1074.5, 1105.0, 1102.0, 1059.0, 1057.25, 1060.75, 1021.5, 1029.0, 1041.0, 1062.75, 1082.0, 1103.5, 1117.0, 1122.0, 1120.0, 1119.25, 1114.25, 1116.0, 1109.0, 902.75, 882.5, 874.25, 866.75, 851.0, 775.5, 529.5, 531.0, 527.75, 528.75, 533.5, 533.5, 499.75, 493.5, 491.25, 486.25, 483.25, 479.0, 476.0, 468.0, 464.0, 457.75, 454.75, 450.0, 0.0, 0.0, 404.5, 404.0, 402.5, 0.0, 277.0, 274.0, 274.75, 273.0, 272.5, 272.0, 0.0, 276.0])
intensivities = np.array([49, 50, 50, 50, 50, 51, 52, 52, 50, 49, 0.0, 0.0, 50, 50, 50, 50, 50, 50, 50, 49, 49, 49, 48, 49, 49, 50, 50, 50, 0.0, 51, 51, 51, 52, 52, 53, 53, 53, 54, 53, 54, 53, 53, 54, 53, 53, 52, 52, 52, 53, 53, 53, 54, 0.0, 0.0, 52, 53, 53, 52, 52, 53, 53, 53, 53, 52, 52, 53, 53, 53, 53, 53, 53, 53, 54, 54, 53, 54, 53, 53, 53, 53, 53, 54, 54, 52, 54, 55, 0.0, 50, 52, 54, 53, 51, 42, 0.0, 44, 46, 47, 47, 47, 48, 48, 49, 52, 53, 51, 51, 25, 31, 32, 20, 31, 30, 0.0, 32, 31, 32, 0.0, 45, 47, 46, 45, 50, 52, 49, 48, 49, 48, 49, 48, 48, 48, 48, 48, 38, 0.0, 49, 48, 48, 48, 46, 47, 49, 48, 48, 48, 49, 47, 45, 49, 50, 46, 51, 48, 45, 32, 48, 47, 24, 0.0, 46, 47, 46, 47, 47, 48, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 46, 46, 45, 45, 46, 46, 46, 44, 43, 45, 34, 46, 44, 45, 44, 45, 42, 45, 44, 45, 45, 45, 45, 46, 46, 46, 46, 46, 47, 46, 47, 46, 46, 46, 46, 47, 47, 46, 46, 47, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 49, 48, 49, 49, 50, 50, 51, 51, 51, 52, 52, 52, 52, 54, 54, 53, 53, 53, 53, 52, 51, 52, 51, 50, 50, 50, 50, 50, 50, 50, 50, 49, 49, 49, 50, 49, 49, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 47, 48, 46, 47, 46, 46, 46, 45, 46, 46, 45, 44, 45, 44, 44, 51, 52, 49, 49, 50, 51, 48, 51, 48, 48, 50, 49, 51, 51, 50, 49, 50, 47, 24, 46, 47, 47, 46, 42, 39, 49, 50, 51, 51, 53, 53, 49, 50, 50, 51, 52, 52, 51, 51, 52, 52, 52, 52, 0.0, 0.0, 54, 53, 53, 0.0, 49, 50, 50, 50, 50, 50, 0.0, 48])    
gf = GlassFilter()
pcs, invalids, candidates = gf.stdv_filter(scan, intensivities, 175, 0, 0, 0)
angles = [a for a in range(0, 360)]
founded_inds = []
window = 15
pca_data = []

for pc in pcs.copy():
    valid, dno_idx = gf.valid_filter(5, 2, 0, 0, 10000, pc, window)
    if not valid:
        pcs.remove(pc)
        continue
    start = pc.points[0].index
    end = pc.points[-1].index
    _, k, b, coords, var_ratio = gf.fitting_filter(scan, [start, dno_idx, end])
    print(f"Fitting: {k} {b} {coords} {var_ratio}")
    if var_ratio < 0.87:
        pcs.remove(pc)
        continue
    for p in pc.points:
        pca_data.append([k, b, coords, var_ratio, [start, dno_idx, end]])
        founded_inds.append(p.index)

sc, oc = gf.patch(pcs, scan)
smoothed = np.convolve(intensivities, np.ones(window)/window, mode='same')
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.scatter(np.radians(angles), sc, color='blue', s=15, label='замеры')
ax.scatter(np.radians(angles), smoothed * 20, color="purple", s=5, label="интенсивности")
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

    if not label:
        ax.scatter(theta_pts, r_pts, color='red', s=25, zorder=5, label="точки для фитинга")
        ax.plot(theta_line, r_line, linestyle='--', linewidth=2, label="линия ")
        label = True
    else: 
        ax.scatter(theta_pts, r_pts, color='red', s=25, zorder=5)
        ax.plot(theta_line, r_line, linestyle='--', linewidth=2)
        

ax.scatter(invalids, [100] * len(invalids), color="red", s=2, marker="o", label='дропауты, выдвеннутые вперед')
ax.scatter(founded_angles, founded_dists, color="green", s=12, marker="o", label='стекла')
plt.title("карта комнаты", pad=20, fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()