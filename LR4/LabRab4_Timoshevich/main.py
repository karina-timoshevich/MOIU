import numpy as np


def dual_simplex_method(c, A, b, B_init):
    m, n = A.shape
    B = list(B_init.copy())

    while True:
        # делаем базисную матрицу и обратную
        AB = A[:, B]
        try:
            AB_inv = np.linalg.inv(AB)
        except np.linalg.LinAlgError:
            print("Матрица AB вырождена")
            return None

        # вектор cB из компонент В
        cB = c[B]

        # базисный допуст план двойственной задачи
        y = cB @ AB_inv

        # псевдоплан
        k_B = AB_inv @ b
        k = np.zeros(n)
        for i, idx in enumerate(B):
            k[idx] = k_B[i]

        # шаг 5
        if np.all(k >= 0):
            return k

        # индекс отриц компоненты псведоплана
        negative_indices = np.where(k_B < 0)[0]
        if len(negative_indices) == 0:
            print("Нет отрицательных компонент, но решение не найдено. Ошибка.")
            return None

        k = negative_indices[0]
        jk = B[k]

        # вычисления mu_j
        delta_y = AB_inv[k, :]
        mu = []
        non_basis = [j for j in range(n) if j not in B]
        for j in non_basis:
            mu_j = delta_y @ A[:, j]
            mu.append((mu_j, j))

        # шаг 8
        if all(mu_j >= 0 for mu_j, _ in mu):
            print("Задача несовместна")
            return None

        # шаг 9
        sigma = []
        for mu_j, j in mu:
            if mu_j < 0:
                numerator = c[j] - A[:, j] @ y
                sigma_j = numerator / mu_j
                sigma.append((sigma_j, j))

        # ищем мин sigma и его индекс
        sigma_min, j0 = min(sigma, key=lambda x: x[0])

        # меняем k-ый базисный индекс на индекс j0
        B[k] = j0


c = np.array([-4, -3, -7, 0, 0])
A = np.array([
    [-2, -1, -4, 1, 0],
    [-2, -2, -2, 0, 1]
])
b = np.array([-1, -1.5])
B_init = np.array([3, 4])

result = dual_simplex_method(c, A, b, B_init)
print("Оптимальный план:", result)


# b⊺y → min A⊺y ⩾ c,