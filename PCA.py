"""
Моя реализация PCA алгоритма
"""

import numpy as np
import matplotlib.pyplot as plt


""" 
Алгоритм:
1. найти средний вектор между точками
2. высчитать матрицу ковариации
3. высчитать собственные значение и собственные векторы 
Собственные значения - это показатель того, сколько информации (дисперсии) удерживает в себе конкретная главная компонента. Чем больше число, тем важнее эта ось для объяснения структуры данных.
Собственные векторы - это направления осей, вдоль которых данные имеют наибольший разброс (вариативность). Первой главной компонентой (PC1) становится собственный вектор с максимальным разбросом данных.
4. отсортировать собсвенные векторы в порядке важности их осей
5. Выбрать первые N векторов (главны компоненты)
6. спроецировать отцентрированные данные на новые главные компоненты
"""

def pca(x: np.array, num_components: int, return_kb: bool = False):
    """Реализация метода главных компонент (PCA)

    Args:
        x (np.array): массив со списками координат точек [[x0, y0], [x1, y1]]
        num_components (int): количество главных компонент
        return_kb (bool): вернуть коэффициенты прямой y=kx+b

    Returns:
        _type_: _description_
    """
    #1
    x_mean = np.mean(x, axis=0)
    x_cent = x - x_mean
    
    #2
    c_mat = np.cov(x_cent, rowvar=False)
    #3
    eigenvals, eigenvecs = np.linalg.eig(c_mat)
    #4
    sorted_ind = np.argsort(eigenvals)[::-1]
    sorted_eigenvals = eigenvals[sorted_ind]
    sorted_eigenvecs = eigenvecs[:, sorted_ind]
    
    #5
    eigenvec_subset = sorted_eigenvecs[:, :num_components]
    
    #6
    x_reduced = np.dot(x_cent, eigenvec_subset)
    if not return_kb:
        return x_reduced, sorted_eigenvals, sorted_eigenvecs, x_mean
    pc1 = sorted_eigenvecs[:, 0]
    v_x, v_y = pc1[0], pc1[1]
    k = v_y / v_x
    mean_x, mean_y = x_mean[0], x_mean[1]
    b = mean_y - k * mean_x
    return x_reduced, sorted_eigenvals, sorted_eigenvecs, x_mean, k, b


def visualize(points):
    #1
    _, _, sorted_eigenvecs, x_mean = pca(points, num_components=1, return_kb=False)
    #2
    pc1 = sorted_eigenvecs[:, 0]
    v_x, v_y = pc1[0], pc1[1]
    # 3
    k = v_y / v_x
    mean_x, mean_y = x_mean[0], x_mean[1]
    b = mean_y - k * mean_x
    print(f"Уравнение прямой: y = {k:.4f} * x + ({b:.4f})")

    plt.scatter(points[:, 0], points[:, 1], color='red', zorder=5, label='Исходные точки')
    x_line = np.linspace(min(points[:, 0]) - 0.5, max(points[:, 0]) + 0.5, 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='blue', linestyle='--', linewidth=2, label=f'PCA линия (y = {k:.2f}x + {b:.2f})')
    plt.scatter(mean_x, mean_y, color='green', marker='X', s=150, zorder=6, label='Центр масс (mean)')
    plt.title('PCA', fontsize=12)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    points = np.array([
        [1.0, 3.1],
        [2.0, 4.9],
        [3.0, 7.2],
        [4.0, 8.8]
    ])
    visualize(points)


"""
Порядок применения:
1. расчиатть pca для точек
2. найти координаты направляющего вектора (первый столбец)
3. рассчитать коэффициенты k b
"""