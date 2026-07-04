import scan
scans = [scan.scan, scan.scan1, scan.scan2]
intensivs = [scan.intesivities, scan.intesivities1, scan.intesivities2]

# voices = [0] * 360

# gf = glass_filter2.GlassFilter()
# for sc in scans:    
#     pcs = gf.stdv_filter(scan.scan, 175, 0, 0, 0)
#     angles = [a for a in range(0, 360)]
#     inds = []
#     for pc in pcs:
#         for p in pc.points:
#             print(f"x: {p.x} y: {p.y} r: {p.range} ind: {p.index}")
#             inds.append(p.index)
#             voices[p.index] += 1
# print(voices)

invalids = []
for i in range(len(scan.scan)):
    if scan.scan[i] <= 0:
        invalids.append(i)
print(invalids)

if invalids is not None:
    prev_inv = invalids[0]
    for i in range(1, len(invalids)):
        
        prev_inv = i