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

# import numpy as np

# # 1. Ваши исходные данные
# data = np.array([9.0, 19.0, 14.0, 0.0, 0.0, 25.0, 31.0, 35.0])
# x = np.arange(len(data))

# # 2. Сглаживание (опционально, убирает случайные скачки)
# window = 3
# smoothed = np.convolve(data, np.ones(window)/window, mode='same')

# # 3. Подгонка полинома 2-й степени: y = ax^2 + bx + c
# a, b, c = np.polyfit(x, data, 2)

# # 4. Находим точку экстремума (вершину параболы)
# vertex_x = -b / (2 * a)

# # 5. Логика определения формы
# print(f"Анализ последовательности:")
# print(f"Коэффициент изгиба (a): {a:.2f}")
# print(f"Точка изгиба (x): {vertex_x:.2f}")

# if a > 0.5 and (0 <= vertex_x <= len(data) - 1):
#     print("Результат: Последовательность имеет U-образную форму (сначала падала, затем выросла).")
# elif a < -0.5 and (0 <= vertex_x <= len(data) - 1):
#     print("Результат: Последовательность имеет перевернутую U-образную форму (холмик).")
# else:
#     # Если явного изгиба внутри диапазона нет, смотрим на общий линейный тренд
#     linear_trend = np.polyfit(x, data, 1)[0]
#     if linear_trend > 1.0:
#         print("Результат: Последовательность преимущественно возрастает.")
#     elif linear_trend < -1.0:
#         print("Результат: Последовательность преимущественно падает.")
#     else:
#         print("Результат: Последовательность стагнирует (плоская).")

cans = [[0, 8], [138, 146], [245, 254], [256, 264], [266, 286], [337, 359]]
for can in range(len(cans)):
    print(cans[can])
    ps, pe = cans[(can-1)%len(cans)]
    s, e = cans[can]
    print(f"ps: {ps} pe: {pe} s: {s} e: {e}")
    print(f"(s-pe)%360: {(s-pe)%360}")
    if (s-pe)%360 < 3:
        print(f"new: {ps} {e}")
#print(sorted(cans, reverse=True))

start = 337
end = 8
total_elements = 360

# Вычисляем, сколько всего шагов нужно сделать (32 шага)
steps = (end - start) % total_elements + 1

for step in range(steps):
    i = (start + step) % total_elements
    print(i)  # Выведет: 337, 338... 359, 0, 1... 8

import numpy as np

data = [48.0, 43.0, 28.0, 15.0, 12.0, 16.0, 23.0, 20.0, 0.0, 0.0, 18.0, 18.0, 17.0, 17.0, 16.0, 0.0, 22.0, 0.0, 53.0, 54.0, 38.0, 22.0, 12.0, 9.0, 19.0, 14.0, 0.0, 0.0, 25.0, 31.0, 35.0, 35.0]

import numpy as np
import scipy.signal as signal

def check_drop_and_recovery_fixed(raw_data):
    if len(raw_data) < 7:
        return "Слишком короткая последовательность"
    
    # 1. Убираем импульсный мусор (нули) медианным фильтром
    clean_data = signal.medfilt(raw_data, kernel_size=3)
    
    # 2. Сглаживаем тренд скользящим средним
    window = 5
    smoothed = np.convolve(clean_data, np.ones(window)/window, mode='same')
    
    # Обрезаем края, чтобы убрать искажения свертки
    valid_zone = smoothed[window//2 : -(window//2)]
    
    # 3. ДЕЛИТЬ НА ЧАСТИ НУЖНО БЫЛО ВОТ ТАК (Индексация по элементам списка):
    chunks = np.array_split(valid_zone, 3)
    mean_start = np.mean(chunks[0])   # Начало (левое плечо)
    mean_middle = np.mean(chunks[1])  # Середина (дно провала)
    mean_end = np.mean(chunks[2])     # Конец (правое плечо)
    
    # Амплитуда изменений
    total_amplitude = np.max(valid_zone) - np.min(valid_zone)
    
    # Защита от плоской стагнации
    if total_amplitude < 10.0:
        return "Стабильное состояние (минимальные колебания)"
    
    # Проверяем условия: провал должен быть ниже краев минимум на 25% от общей амплитуды
    is_dropping = (mean_start - mean_middle) > (total_amplitude * 0.25)
    is_recovering = (mean_end - mean_middle) > (total_amplitude * 0.25)
        
    if is_dropping and is_recovering:
        return f"ПОДТВЕРЖДЕНО: Последовательность упала и выросла обратно! (Амплитуда: {total_amplitude:.1f})"
    elif is_dropping:
        return "Только падение (сигнал ушел вниз)"
    elif is_recovering:
        return "Только рост (сигнал вышел из низины)"
    
    return "Сложный или хаотичный тренд"

# Проверка вашей последовательности scrap
scrap = [36.0, 23.0, 0.0, 0.0, 10.0, 12.0, 0.0, 13.0, 11.0, 11.0, 10.0, 18.0, 27.0, 40.0, 37.0, 27.0, 14.0, 0.0, 0.0, 9.0, 13.0, 15.0, 15.0, 0.0, 16.0, 0.0, 41.0]
print("Вердикт для scrap:", check_drop_and_recovery_fixed(scrap))

from algorithms import median_filter

k_size = 3
custom_result = median_filter(data, kernel_size=k_size)

print("Исходный массив: ", data)
print("фильтр:", custom_result)