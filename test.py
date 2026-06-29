# posl = [50, 100, 100, 25, 65, 75, 50, 0, 50, 80, 100, 90, 100, 100]
# posl = [100, 80, 60, 60, 90, 0, 100, 85, 110, 0, 110, 0, 90, 110, 80, 100]
posl = [52, 52, 52, 53, 53, 53, 53, 53, 53, 52, 0.0, 36, 37, 37, 0.0, 49, 52, 49, 39, 37, 34, 54, 53, 53, 54, 52, 53, 53, 53, 53, 53, 52, 22, 20, 52, 52, 53, 53, 53, 52, 52, 52, 52, 52, 52, 52, 52, 52, 51, 51, 52, 52, 52, 51, 52, 51, 51, 51, 51, 50, 50, 49, 50, 50, 50, 49, 49, 48, 44, 0.0, 50, 51, 53, 53, 51, 51, 50, 48, 35, 0.0, 0.0, 40, 42, 0.0, 0.0, 47, 47, 47, 48, 26, 0.0, 0.0, 46, 42, 0.0, 0.0, 17, 22, 24, 26, 25, 24, 25, 26, 24, 26, 24, 24, 24, 24, 23, 23, 23, 22, 22, 20, 21, 20, 20, 20, 20, 20, 19, 18, 16, 15, 14, 14, 19, 47, 52, 46, 46, 46, 46, 44, 45, 44, 43, 43, 43, 43, 43, 43, 44, 42, 42, 50, 44, 47, 48, 48, 46, 0.0, 37, 47, 47, 45, 43, 43, 39, 0.0, 46, 39, 40, 51, 49, 45, 44, 42, 33, 36, 0.0, 39, 37, 43, 44,
        44, 46, 46, 47, 45, 45, 42, 45, 45, 36, 35, 36, 0.0, 0.0, 34, 28, 0.0, 44, 0.0, 50, 50, 50, 50, 51, 52, 51, 50, 50, 50, 50, 50, 51, 52, 51, 51, 51, 50, 49, 49, 50, 50, 50, 50, 50, 50, 50, 49, 49, 48, 48, 48, 48, 40, 22, 44, 51, 53, 55, 55, 54, 55, 52, 49, 46, 48, 50, 53, 53, 52, 52, 51, 49, 44, 0.0, 0.0, 0.0, 26, 32, 32, 32, 30, 32, 31, 11, 0.0, 0.0, 0.0, 21, 24, 1, 2, 18, 18, 1, 21, 19, 20, 19, 19, 13, 0.0, 21, 18, 19, 21, 17, 0.0, 29, 32, 0.0, 15, 11, 0.0, 0.0, 10, 0.0, 8, 9, 6, 0.0, 0.0, 7, 5, 0.0, 0.0, 0.0, 8, 6, 6, 0.0, 7, 7, 6, 0.0, 5, 8, 5, 0.0, 42, 42, 42, 41, 42, 42, 42, 42, 42, 43, 43, 43, 42, 42, 42, 38, 32, 45, 28, 0.0, 47, 0.0, 49, 30, 7, 5, 0.0, 49, 51, 50, 50, 51, 51, 47, 0.0, 0.0, 51, 51, 51, 52, 52, 52, 52, 52, 53]
peaks = []
prev = posl[0]
s = e = 0
for i in range(1, len(posl)):
    if posl[i] >= prev and prev != 0:
        e = i
    else:
        if e > s:
            peaks.append([s, e])
        s = i
    prev = posl[i]
if e > s:
    peaks.append([s, e])
print(peaks)
for i in peaks:
    for j in range(i[0], i[1] + 1):
        print(posl[j])
    print()


s = e = 0
prev = posl[0]
downs = []
for i in range(1, len(posl)):
    if posl[i] < prev and posl[i] != 0:
        e = i
    else:
        if e > s:
            downs.append([s, e])
        s = i
    prev = posl[i]
if e > s:
    downs.append([s, e])
print(downs)
for i in downs:
    for j in range(i[0], i[1] + 1):
        print(posl[j])
    print()


# склеивание кусков в одно
gaps = []


def clue(peaks: list, downs: list):
    if len(peaks) == 0 or len(downs) == 0:
        return peaks, downs
    ind_u = peaks[0][0]
    ind_d = downs[0][0]
    trash = []
    i = 0
    j = 0
    if ind_u < ind_d:
        while i < len(peaks) and j < len(downs):
            end_ind = peaks[i][1]
            start_ind = downs[j][0]
            if end_ind == start_ind:
                i += 1
                j += 1
            elif end_ind < start_ind:
                trash.append(peaks.pop(i))
            else:
                trash.append(downs.pop(j))
    else:
        while i < len(peaks) and j < len(downs):
            end_ind = downs[j][1]
            start_ind = peaks[i][0]
            if end_ind == start_ind:
                i += 1
                j += 1
            elif end_ind < start_ind:
                trash.append(downs.pop(j))
            else:
                trash.append(peaks.pop(i))

    if len(peaks) > len(downs):
        peaks.pop()
    else:
        downs.pop()
    print(f"Итоговые длины: возрастания {len(peaks)}, падения {len(downs)}")
    return peaks, downs


clue(peaks=peaks, downs=downs)
