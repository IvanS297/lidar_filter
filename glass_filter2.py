import scan
import numpy as np
import matplotlib.pyplot as plt


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
        """Алгоритм 1: фильтр скана по скользящим окнас и STDV (СКО)

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
        # print(candidates)
        return pcs, invalids
    
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
pcs, invalids = gf.stdv_filter(scan.scan, 175, 0, 0, 0)
angles = [a for a in range(0, 360)]
inds = []
for pc in pcs:
    for p in pc.points:
        print(f"x: {p.x} y: {p.y} r: {p.range} ind: {p.index}")
        inds.append(p.index)
sc, oc = gf.patch(pcs, scan.scan)
# inds = gf.accum_scans([scan.scan, scan.scan1, scan.scan2], 0, 0, 0)

# sc = np.array([
#     2147.25, 2153.5, 2147.0, 2146.5, 2146.5, 0.0, 2137.5, 2140.5, 2149.0, 2161.0, 2164.0, 2166.0, 0.0, 693.5, 0.0, 635.0, 645.5, 648.0, 646.5, 589.0, 574.5, 567.0, 566.5, 564.5, 564.0, 564.0, 565.0, 567.25, 567.75, 572.0, 579.5, 582.75, 587.75, 607.25, 618.25, 0.0, 684.75, 636.25, 631.25, 620.5, 570.0, 584.5, 573.0, 557.0, 558.0, 564.0, 567.5, 574.25, 579.5, 584.5, 595.0, 604.0, 609.25, 621.5, 626.0, 638.0, 650.0, 655.5, 667.5, 675.5, 689.25, 698.75, 716.5, 724.5, 738.5, 729.5, 730.75, 737.0, 743.75, 775.0, 775.5, 772.0, 0.0, 0.0, 1173.5, 1164.0, 1159.5, 1154.5, 771.0, 766.0, 761.0, 761.0, 772.5, 779.25, 803.0, 0.0, 1108.0, 831.0, 813.25, 809.5, 809.0, 813.0, 817.5, 828.0, 1093.5, 1097.0, 1096.5, 1099.0, 1097.5, 1098.5, 1099.75, 1103.75, 1104.5, 1110.5, 1111.0, 1114.0, 1109.75, 1107.5, 1147.5, 1120.5, 776.75, 784.0, 1103.5, 1128.0, 1176.5, 1190.0, 1200.0, 1202.75, 1212.5, 1219.0, 1221.75, 1219.0, 1212.75, 951.5, 957.75, 964.0, 964.0, 1017.0, 1024.0, 1026.0, 978.75, 1314.5, 995.5, 986.0, 973.5, 966.5, 960.0, 139.0, 139.5, 137.75, 136.5, 133.5, 132.25, 130.0, 129.0, 128.0, 128.75, 131.25, 131.5, 135.75, 138.75, 140.25, 138.75, 142.0, 1163.75, 1165.0, 1163.0, 1155.5, 1150.0, 1145.5, 1133.75, 1129.75, 1122.5, 0.0, 838.5, 836.25, 836.75, 836.5, 834.25, 833.5, 834.0, 833.0, 833.0, 835.5, 835.0, 835.5, 836.5, 835.0, 832.75, 835.5, 836.75, 0.0, 1084.0, 1087.5, 1095.0, 1094.25, 1096.0, 1098.5, 1105.5, 1105.5, 1108.0, 1114.5, 1121.5, 1121.75, 1121.75, 1129.0, 1128.5, 1120.0, 0.0, 932.75, 898.5, 882.0, 851.0, 840.5, 824.0, 834.25, 0.0, 646.0, 609.0, 0.0, 648.5, 642.5, 593.75, 585.25, 552.5, 548.25, 557.75, 564.0, 572.25, 0.0, 619.0, 612.0, 611.5, 624.5, 624.0, 614.0, 604.75, 606.5, 0.0, 173.5, 170.0, 166.25, 163.5, 161.75, 161.25, 169.5, 171.0, 177.0, 0.0, 186.75, 683.0, 682.75, 686.5, 695.0, 735.5, 744.25, 731.5, 724.5, 714.75, 671.75, 662.0, 655.0, 651.25, 645.0, 642.5, 635.5, 632.5, 626.5, 625.0, 621.5, 618.75, 614.0, 611.0, 604.25, 0.0, 543.25, 533.75, 523.5, 512.0, 510.0, 504.25, 500.25, 490.5, 486.0, 481.25, 477.0, 465.75, 0.0, 250.5, 250.5, 249.25, 246.75, 246.5, 245.5, 244.75, 246.0, 248.25, 249.25, 251.5, 253.5, 253.0, 256.0, 262.5, 265.5, 268.75, 271.0, 273.75, 275.25, 281.5, 284.5, 287.0, 294.25, 298.75, 303.5, 307.0, 311.5, 304.75, 323.5, 328.5, 340.25, 344.0, 353.25, 358.25, 370.0, 0.0, 393.0, 0.0, 425.0, 435.25, 441.25, 448.25, 467.0, 476.5, 488.0, 537.0, 565.5, 577.0, 569.25, 560.25, 550.75, 557.75, 573.5, 574.0, 583.75, 582.5, 600.25, 0.0, 699.5, 0.0, 858.0, 878.5, 895.5, 940.5, 965.0, 991.5, 998.5, 986.5, 983.75, 997.0, 0.0, 0.0, 0.0, 4614.5, 4468.0, 4430.0, 4489.5, 0.0, 0.0, 2193.0, 2145.75
# ])
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.scatter(np.radians(angles), sc, color='blue', s=15, label='замеры')
ax.scatter(0, 0, color='red', s=120, marker='*', label='лидар')
founded_angles = np.radians(np.array(angles)[oc])
founded_dists = np.array(sc)[oc]
# ax.scatter(invalids, [100] * len(invalids), color="green", s=2, marker="o", label='стекла')
ax.scatter(founded_angles, founded_dists, color="green", s=12, marker="o", label='стекла')
plt.title("карта комнаты", pad=20, fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()