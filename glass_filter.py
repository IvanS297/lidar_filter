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
        s = e = 0
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
        if not 0 in intesiv_scrap:
            #print(f"no dropouts")
            return False, None

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
            # win_frame_left = 0
            # ind = (s - 1)%360
            # vars = []
            # while new_scan[s] > win_frame_left:
            #     win_frame_left = new_scan[ind]
            #     vars.append(new_scan[ind])
            #     if (s - ind) % 360 > 12:
            #         win_frame_left = max(vars)
            #         break
            #     ind = (ind - 1) % 360
            # win_frame_right = 0
            # ind = (e + 1)%360
            # #print(f"vars left: {vars}")
            # vars = []
            # while scan[e] > win_frame_right:
            #     win_frame_right = new_scan[ind]
            #     vars.append(new_scan[ind])
            #     if (ind - e) % 360 > 12:
            #         win_frame_right = max(vars)
            #         break
            #     ind = (ind + 1) % 360

            
            if e >= s:
                num_elements = e - s + 1
            else:
                # Переход через ноль (например, от 337 до 8 при длине 360)
                num_elements = (len(pc.points) - s) + (e + 1)
            indices = np.array([(s + i) % len(pc.points) for i in range(len(new_scan))])
            theta = np.radians(indices)
            denominator = np.sin(theta) - k * np.cos(theta)
            r = b / denominator
            r_correct = np.abs(r)
            
            for i in range(num_elements):
                idx = indices[i]
                new_scan[idx] = r_correct[i]
        return new_scan, occupied

scan = np.array([822.5, 823.25, 826.5, 828.5, 829.5, 830.75, 831.25, 836.0, 842.5, 842.0, 848.0, 850.0, 851.0, 858.5, 862.0, 868.75, 874.0, 882.0, 888.0, 893.0, 898.5, 903.5, 916.0, 923.0, 928.75, 936.25, 949.5, 954.5, 965.75, 976.75, 982.5, 998.0, 1006.0, 1025.0, 1038.5, 1051.5, 1070.75, 1081.5, 1102.25, 1125.5, 1137.5, 1162.25, 1191.0, 1204.5, 1232.5, 1247.0, 1278.75, 1296.0, 1331.25, 1371.75, 1392.5, 1436.0, 1458.0, 1504.5, 1558.25, 1584.75, 1643.5, 1707.5, 1743.5, 1822.0, 1863.5, 1952.75, 2050.5, 2107.0, 2221.0, 2282.5, 2420.0, 2578.0, 0.0, 0.0, 0.0, 3141.75, 0.0, 2900.5, 2937.5, 0.0, 2335.25, 2298.5, 2299.0, 2318.0, 354.25, 354.0, 354.25, 357.0, 362.0, 373.0, 379.0, 386.0, 378.5, 370.5, 371.75, 374.0, 377.5, 381.5, 384.5, 384.75, 385.0, 386.75, 390.75, 393.25, 402.0, 0.0, 370.5, 367.0, 364.5, 361.5, 360.0, 360.5, 361.0, 365.25, 378.0, 0.0, 410.5, 414.0, 417.5, 429.0, 436.0, 450.5, 456.0, 456.0, 458.75, 469.5, 479.0, 496.0, 498.5, 490.5, 492.0, 494.0, 500.25, 504.75, 513.5, 519.25, 515.0, 521.5, 523.0, 523.5, 522.25, 523.0, 524.75, 528.25, 533.5, 536.5, 545.5, 552.25, 560.25, 642.0, 0.0, 698.25, 711.75, 743.0, 783.5, 0.0, 885.0, 941.5, 958.5, 964.5, 972.25, 0.0, 1205.0, 0.0, 1335.5, 1357.0, 0.0, 1782.0, 1827.5, 1807.5, 1805.5, 1792.5, 1803.0, 1871.5, 1895.0, 1901.25, 1901.75, 1913.5, 0.0, 1719.5, 1693.0, 1671.5, 1598.0, 0.0, 1462.5, 0.0, 1356.75, 1336.75, 1317.25, 1298.25, 1290.5, 1261.25, 1243.5, 1151.0, 1140.5, 1138.0, 1070.5, 991.0, 0.0, 794.0, 789.5, 781.5, 774.0, 687.25, 0.0, 563.5, 568.0, 568.5, 574.0, 578.0, 582.0, 586.5, 600.0, 604.5, 612.5, 622.25, 623.5, 629.5, 630.5, 627.25, 628.0, 631.5, 633.5, 633.25, 633.25, 634.0, 631.5, 630.5, 635.5, 638.5, 647.5, 652.5, 907.0, 921.5, 946.5, 960.25, 929.5, 911.0, 903.25, 892.5, 886.0, 876.25, 871.5, 862.0, 857.5, 845.5, 840.25, 833.0, 829.0, 822.75, 819.25, 814.0, 808.0, 805.0, 797.25, 796.0, 792.25, 789.25, 787.5, 786.5, 781.25, 780.25, 780.5, 779.0, 779.0, 776.5, 777.5, 780.5, 780.0, 778.0, 779.25, 778.5, 778.0, 777.75, 779.75, 798.0, 790.5, 0.0, 0.0, 0.0, 617.5, 586.25, 579.75, 585.75, 589.75, 593.5, 0.0, 556.75, 581.0, 607.0, 609.5, 0.0, 545.25, 0.0, 475.25, 452.5, 441.25, 436.5, 426.5, 422.0, 413.5, 410.0, 399.5, 396.0, 388.75, 384.25, 380.0, 373.5, 362.25, 0.0, 322.25, 319.0, 317.5, 316.0, 316.5, 315.5, 317.75, 319.0, 322.0, 325.5, 333.5, 335.0, 343.0, 344.5, 351.75, 355.25, 0.0, 0.0, 0.0, 345.75, 340.25, 342.0, 343.5, 347.75, 358.75, 0.0, 0.0, 429.25, 444.0, 450.5, 462.25, 0.0, 542.0, 541.5, 541.25, 541.75, 552.25, 0.0, 600.75, 609.5, 603.5, 576.5, 577.5, 818.5, 829.25, 826.5, 823.25, 822.75, 818.5, 818.5, 818.5, 0.0, 820.5, 820.0])
intensivities = np.array([50, 49, 50, 50, 49, 49, 49, 49, 49, 49, 50, 50, 49, 50, 50, 50, 50, 51, 50, 50, 50, 50, 50, 50, 50, 49, 49, 47, 47, 47, 47, 46, 47, 47, 46, 47, 46, 47, 47, 46, 47, 46, 47, 46, 46, 46, 46, 45, 45, 45, 45, 45, 44, 45, 44, 43, 43, 42, 41, 41, 41, 39, 39, 38, 37, 37, 35, 32, 0.0, 0.0, 0.0, 39, 0.0, 32, 13, 0.0, 36, 37, 34, 29, 50, 53, 53, 52, 50, 52, 54, 55, 53, 52, 52, 52, 52, 52, 53, 51, 52, 52, 52, 52, 55, 0.0, 52, 51, 52, 52, 52, 52, 52, 50, 45, 0.0, 52, 52, 50, 47, 49, 52, 53, 52, 51, 47, 46, 53, 53, 51, 51, 50, 50, 50, 52, 52, 50, 50, 51, 50, 50, 49, 49, 50, 49, 49, 47, 47, 43, 44, 0.0, 45, 46, 44, 43, 0.0, 42, 49, 48, 47, 43, 0.0, 41, 0.0, 43, 43, 0.0, 44, 45, 46, 44, 43, 42, 45, 48, 44, 45, 42, 0.0, 47, 44, 40, 37, 0.0, 45, 0.0, 48, 47, 46, 47, 47, 46, 49, 44, 45, 46, 45, 40, 0.0, 45, 45, 43, 45, 44, 0.0, 46, 46, 43, 46, 46, 44, 46, 44, 44, 44, 46, 46, 47, 46, 46, 46, 45, 46, 45, 45, 45, 45, 44, 46, 46, 46, 26, 46, 46, 48, 51, 46, 47, 47, 46, 46, 47, 47, 47, 46, 46, 47, 47, 47, 47, 48, 48, 47, 48, 47, 48, 47, 47, 48, 47, 47, 48, 48, 49, 49, 50, 51, 52, 53, 51, 52, 52, 51, 49, 50, 53, 51, 0.0, 0.0, 0.0, 40, 45, 49, 51, 51, 50, 0.0, 49, 51, 56, 55, 0.0, 43, 0.0, 39, 46, 50, 50, 51, 50, 51, 50, 51, 50, 50, 50, 51, 48, 46, 0.0, 49, 51, 52, 53, 52, 52, 52, 52, 51, 50, 50, 51, 51, 51, 51, 50, 0.0, 0.0, 0.0, 52, 52, 53, 53, 51, 46, 0.0, 0.0, 46, 48, 49, 45, 0.0, 50, 49, 49, 49, 44, 0.0, 55, 56, 53, 51, 41, 51, 52, 52, 50, 51, 50, 50, 50, 0.0, 50, 50])    
gf = GlassFilter()
pcs, invalids, candidates = gf.stdv_filter(scan, intensivities, 175, 0, 0, 0)
angles = [a for a in range(0, 360)]
founded_inds = []
window = 15
pca_data = []

for pc in pcs.copy():
    valid, dno_idx = gf.valid_filter(20, 2, 0, 0, 10000, pc, window)
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