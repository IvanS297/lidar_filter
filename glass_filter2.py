import scan
import numpy as np
import matplotlib.pyplot as plt
from PCA import *
from plot import *


class Beam:
    def __init__(self, range, index):
        self.range = range
        self.index = index

class Point:
    def __init__(self, x, y, index, range = None):
        self.x = x
        self.y = y
        self.index = index
        self.range = range
        self.beam = Beam(self.range, index)

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
    
    def stdv_filter(self, scan: list, threshold: float, x: float, y: float, theta: float):
        """Алгоритм 1: фильтр скана по скользящим окнам и STDV (СКО)

        Args:
            scan (list | np.array): скан лидара
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
                mbeam = Beam(r, j)
                mbeams.append(mbeam)
        
        # склейка лучей в обрывки
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
                points = []
                for j in range(s, e):
                    x = coords[j][0]
                    y = coords[j][1]
                    p = Point(x, y, j, scan[j])
                    points.append(p)
                pc = PointCloud(points)
                pcs.append(pc)
                s = ind
                e = ind
            prev_ind = ind
        print(candidates)
        return pcs, invalids, candidates
    
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
            vars = []
            while scan[e] > win_frame_right:
                win_frame_right = scan[ind]
                vars.append(scan[ind])
                if (ind - e) % 360 > 12:
                    win_frame_right = max(vars)
                    break
                ind = (ind + 1) % 360
            frame = win_frame_left if win_frame_right < win_frame_left else win_frame_right
            for p in range(s, e):
                scan[p] = frame
                occupied[p] = 1
        return scan, occupied
    
gf = GlassFilter()
pcs, invalids, candidates = gf.stdv_filter(scan.scan, 175, 0, 0, 0)
angles = [a for a in range(0, 360)]
inds = []
for pc in pcs:
    for p in pc.points:
        # print(f"x: {p.x} y: {p.y} r: {p.range} ind: {p.index}")
        inds.append(p.index)
#sc, oc = gf.patch(pcs, scan.scan)
# inds = gf.accum_scans([scan.scan, scan.scan1, scan.scan2], 0, 0, 0)
vecs, x_means, ks, bs, coords = gf.fitting_filter(scan.scan, candidates)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.scatter(np.radians(angles), scan.scan, color='blue', s=15, label='замеры')
ax.scatter(0, 0, color='red', s=120, marker='*', label='лидар')
founded_angles = np.radians(np.array(angles)[inds])
founded_dists = np.array(scan.scan)[inds]
ax.scatter(invalids, [100] * len(invalids), color="red", s=2, marker="o", label='дропауты, выдвеннутые вперед')
ax.scatter(founded_angles, founded_dists, color="green", s=12, marker="o", label='стекла')
plt.title("карта комнаты", pad=20, fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# plot_polar_fit(sc, angles, invalids, inds, coords, ks, bs, x_means)