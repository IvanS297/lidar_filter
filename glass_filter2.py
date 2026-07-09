import scan
import numpy as np
import matplotlib.pyplot as plt
from algorithms import *
from plot import *


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
    
    def valid_filter(self, min_points: int, min_amp: int, x: float, y: float, max_chng: int, pc: PointCloud, window: int) -> bool:
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
        # фильтруем сначала по ширине последовательности, потому что дальше есть функции, которые работают с итераторами в списках. МОжет быть IndexErro или какой-нибудь дргуой краш
        #print(f"Последовательность: {pc.points[0].index} {pc.points[-1].index}")
        # ширина
        if len(pc.points) < min_points:
            #print(f"len is invalid: {len(pc.points)}")
            return False

        # новый критерий: дропауты интенсивности
        intesiv_scrap = [pc.points[i].beam.intensiv for i in range(len(pc.points))]
        if not 0 in intesiv_scrap:
            #print(f"no dropouts")
            return False

        # амплитуда:
        # медианный фильтр
        median = median_filter(signal=intesiv_scrap, kernel_size=3)
        smoothed = np.convolve(median, np.ones(window)/window, mode='same') # сглаживание
        # дно
        local_dno_idx = np.argmin(smoothed[1:-1])
        dno_idx = local_dno_idx + 1 
        dno = smoothed[dno_idx] 
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
            return False

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
            return False

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
            print(f"последовательность упала и выросла обратно, старт: {pc.points[0].index}, дно: {pc.points[0].index + dno_idx}, конец: {pc.points[-1].index})")
            return True
        return False
    
    def fitting_filter(self, scan: list[float], inds: list[list[int]]):
        coordinates = self.scan_to_points(0, 0, 0, scan)
        vecs = []
        x_means = []
        ks = []
        bs = []
        coords1 = []
        for scrap in inds:
            coords = [coordinates[r] for r in range(scrap[0], scrap[1])]
            coords1.append(coords)
            _, _, sorted_eigenvecs, x_mean, k, b = pca(coords, num_components=1, return_kb=True)
            print(f"Sorted eigen vectors: {sorted_eigenvecs} x mean: {x_mean} k: {k} b: {b}")
            vecs.append(sorted_eigenvecs)
            x_means.append(x_mean)
            ks.append(k)
            bs.append(b)
        return vecs, x_means, ks, bs, coords1
    
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
        occupied = [0]*len(scan)
        for pc in pcog:
            s, e = pc.points[0].index, pc.points[-1].index
            win_frame_left = 0
            ind = (s - 1)%360
            vars = []
            while scan[s] > win_frame_left:
                win_frame_left = scan[ind]
                vars.append(scan[ind])
                if (s - ind) % 360 > 12:
                    win_frame_left = max(vars)
                    break
                ind = (ind - 1) % 360
            win_frame_right = 0
            ind = (e + 1)%360
            #print(f"vars left: {vars}")
            vars = []
            while scan[e] > win_frame_right:
                win_frame_right = scan[ind]
                vars.append(scan[ind])
                if (ind - e) % 360 > 12:
                    win_frame_right = max(vars)
                    break
                ind = (ind + 1) % 360
            #print(f"vars right: {vars}")
            #frame = win_frame_left if win_frame_right > win_frame_left else win_frame_right
            frame = (win_frame_left + win_frame_right) / 2 # так работает намного лучше
            steps = (e - s) % 360 + 1
            #print(f"lframe: {win_frame_left} r_frame: {win_frame_right} s: {s} e: {e} points count: {steps}")
            for step in range(steps):
                p = (s + step) % 360
                scan[p] = frame
                occupied[p] = 1
        return scan, occupied
    
gf = GlassFilter()
pcs, invalids, candidates = gf.stdv_filter(scan.scan1, scan.intesivities1, 175, 0, 0, 0)
angles = [a for a in range(0, 360)]
inds = []
window = 15

for pc in pcs.copy():
    valid = gf.valid_filter(20, 2, 0, 0, 10000, pc, window)
    if not valid:
        pcs.remove(pc)
        continue
    for p in pc.points:
        inds.append(p.index)

sc, oc = gf.patch(pcs, scan.scan1)
# inds = gf.accum_scans([scan.scan, scan.scan1, scan.scan2], 0, 0, 0)
smoothed = np.convolve(scan.intesivities1, np.ones(window)/window, mode='same')
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.scatter(np.radians(angles), sc, color='blue', s=15, label='замеры')
ax.scatter(np.radians(angles), smoothed * 20, color="purple", s=5, label="интенсивности")
ax.scatter(0, 0, color='red', s=120, marker='*', label='лидар')
founded_angles = np.radians(np.array(angles)[inds])
founded_dists = np.array(sc)[inds]
ax.scatter(invalids, [100] * len(invalids), color="red", s=2, marker="o", label='дропауты, выдвеннутые вперед')
ax.scatter(founded_angles, founded_dists, color="green", s=12, marker="o", label='стекла')
plt.title("карта комнаты", pad=20, fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()