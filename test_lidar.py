#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Чтение одного полного оборота с RPLIDAR C1 (Windows 10).

Что делает скрипт:
  1. Подключается к лидару по COM-порту на скорости 460800 (дефолт для C1).
  2. Печатает служебную информацию об устройстве (модель, прошивка, серийник, здоровье).
  3. Собирает ОДИН полный оборот (360°) и формирует массивы:
       - ranges        : расстояния (мм) по каждому измерению
       - intensities   : интенсивность / quality (сила сигнала) по каждому измерению
       - angles        : углы (градусы) по каждому измерению
       - ranges_360    : расстояние по каждому целому градусу (как ROS LaserScan, длина 360)
       - intens_360    : интенсивность по каждому целому градусу (длина 360)
       - raw           : сырые кортежи (quality, angle, distance) — вся информация с оборота
  4. Выводит все массивы в консоль.

Установка зависимостей:
    pip install rplidar-roboticia matplotlib

Запуск:
    python rplidar_c1_full_scan.py            # порт по умолчанию (PORT ниже)
    python rplidar_c1_full_scan.py COM5       # указать порт вручную
"""

import sys
import matplotlib.pyplot as plt
from rplidar import RPLidar, RPLidarException
import numpy as np

# ---- НАСТРОЙКИ -------------------------------------------------------------
PORT = "COM16"        # <-- ПОМЕНЯЙ на свой порт (см. Диспетчер устройств -> Порты COM и LPT)
BAUDRATE = 460800    # для RPLIDAR C1 это значение обязательно
TIMEOUT = 3
# Пропустить первые N оборотов (они часто неполные при старте мотора)
SKIP_FIRST_SCANS = 2
# ---------------------------------------------------------------------------


def list_ports():
    """Показать доступные COM-порты, чтобы было проще найти лидар."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if ports:
            print("Доступные COM-порты:")
            for p in ports:
                print(f"   {p.device}  -  {p.description}")
        else:
            print("COM-порты не найдены. Проверь кабель и драйвер CP210x.")
    except Exception as e:
        print(f"Не удалось получить список портов: {e}")
    print()


def print_device_info(lidar):
    """Служебная информация. Обёрнута в try/except: на C1 иногда капризничает,
    но это не мешает основному сканированию."""
    try:
        info = lidar.get_info()
        print("=== Информация об устройстве ===")
        for k, v in info.items():
            print(f"   {k}: {v}")
    except RPLidarException as e:
        print(f"get_info() недоступен: {e}")

    try:
        health = lidar.get_health()
        print(f"=== Здоровье ===\n   status={health[0]}, error_code={health[1]}")
    except RPLidarException as e:
        print(f"get_health() недоступен: {e}")
    print()


def collect_one_full_scan(lidar):
    """Возвращает список измерений [(quality, angle, distance), ...] за один полный оборот."""
    scans_seen = 0
    # iter_scans сам определяет границу оборота по флагу new_scan.
    # max_buf_meas повышен, т.к. C1 на 460800 выдаёт много точек — иначе переполнение буфера.
    for scan in lidar.iter_scans(max_buf_meas=5000, min_len=5):
        scans_seen += 1
        if scans_seen <= SKIP_FIRST_SCANS:
            continue
        return scan
    return []


def plot_scan(angles, ranges, intensities):
    """Два графика: ranges по углу и intensities по углу.
    Данные сортируем по углу, чтобы линия шла слева направо."""
    # Сортируем по возрастанию угла
    order = sorted(range(len(angles)), key=lambda i: angles[i])
    a = [angles[i] for i in order]
    r = [ranges[i] for i in order]
    inten = [intensities[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle("RPLIDAR C1 — один полный оборот")

    # --- График 1: расстояние ---
    ax1.plot(a, r, color="tab:blue", linewidth=0.8, marker=".", markersize=3)
    ax1.set_title("Расстояние (ranges)")
    ax1.set_xlabel("Угол, градусы")
    ax1.set_ylabel("Расстояние, мм")
    ax1.set_xlim(0, 360)
    ax1.grid(True, alpha=0.3)

    # --- График 2: интенсивность ---
    ax2.plot(a, inten, color="tab:red", linewidth=0.8, marker=".", markersize=3)
    ax2.set_title("Интенсивность (intensities)")
    ax2.set_xlabel("Угол, градусы")
    ax2.set_ylabel("Уровень интенсивности (quality)")
    ax2.set_xlim(0, 360)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    fig, ax3 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.scatter(np.radians(angles), ranges, color='blue', s=15, label='замеры')
    ax3.scatter(0, 0, color='red', s=120, marker='*', label='лидар')
    plt.title("карта комнаты", pad=20, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    global PORT
    if len(sys.argv) > 1:
        PORT = sys.argv[1]

    list_ports()
    print(f"Подключаюсь к {PORT} @ {BAUDRATE}...\n")

    lidar = RPLidar(PORT, baudrate=BAUDRATE, timeout=TIMEOUT)

    try:
        lidar.clean_input()        # на всякий случай чистим буфер перед командами
        print_device_info(lidar)

        print("Собираю один полный оборот...\n")
        scan = collect_one_full_scan(lidar)

        if not scan:
            print("Не удалось получить полный оборот. Проверь питание и что мотор крутится.")
            return

        # --- Разбираем оборот на массивы ---
        # Каждое измерение: (quality, angle_deg, distance_mm)
        intensities = [m[0] for m in scan]   # сила сигнала (аналог intensities в ROS)
        angles      = [m[1] for m in scan]   # угол в градусах 0..360
        ranges      = [m[2] for m in scan]   # расстояние в мм

        # Версия "по градусам" (как LaserScan: индекс = градус 0..359)
        ranges_360 = [0.0] * 360
        intens_360 = [0.0] * 360
        for q, a, d in scan:
            idx = int(a) % 360
            ranges_360[idx] = d
            intens_360[idx] = q

        # --- Вывод в консоль ---
        print(f"Точек в обороте: {len(scan)}\n")

        print("=== RAW (вся информация: quality, angle°, distance_mm) ===")
        print(scan)
        print()

        print("=== ANGLES (градусы) ===")
        print(angles)
        print()

        print("=== RANGES (расстояния, мм) ===")
        print(ranges)
        print()

        print("=== INTENSITIES (интенсивность / quality) ===")
        print(intensities)
        print()

        print("=== RANGES_360 (расстояние по каждому целому градусу, мм; 0 = нет данных) ===")
        print(ranges_360)
        print()

        print("=== INTENSITIES_360 (интенсивность по каждому целому градусу) ===")
        print(intens_360)
        print()

        # Небольшая сводка
        valid = [d for d in ranges if d > 0]
        if valid:
            print("=== Сводка ===")
            print(f"   мин. расстояние: {min(valid):.1f} мм")
            print(f"   макс. расстояние: {max(valid):.1f} мм")
            print(f"   среднее: {sum(valid) / len(valid):.1f} мм")
            print(f"   диапазон углов: {min(angles):.1f}° .. {max(angles):.1f}°")

        # --- Графики ---
        print("\nСтрою графики (закрой окно, чтобы завершить программу)...")
        plot_scan(angles, ranges, intensities)

    except RPLidarException as e:
        print(f"Ошибка RPLidar: {e}")
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
    finally:
        # Корректное завершение, иначе мотор продолжит крутиться
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
        except Exception:
            pass
        print("\nЛидар остановлен и отключён.")


if __name__ == "__main__":
    main()