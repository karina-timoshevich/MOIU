import numpy as np


def simplex(c, A, b, basis):
    rows, cols = A.shape
    step = 0

    while True:
        step += 1
        print(step)
        A_basis = A[:, basis]
        A_basis_inv = np.linalg.inv(A_basis)
        c_basis = c[basis]
        potentials = np.dot(c_basis, A_basis_inv)

        # определяем оценки
        deltas = np.dot(potentials, A) - c

        if np.all(deltas >= 0):
            return b

        # тщем индекс первой отрицательной оценки
        entering_var = np.argmin(deltas)

        # рассчитываем вектор направлений z
        direction = np.dot(A_basis_inv,
                           A[:, entering_var])  # пр как будет вести себя система, если мы добавим новый товар

        # опр вектор θ
        theta_vals = np.full(rows, np.inf)
        valid_indices = np.where(direction > 0)[0]  # извлекаем только индексы эл direction > 0
        theta_vals[valid_indices] = np.take(b, basis[valid_indices]) / direction[valid_indices]

        min_theta = np.min(theta_vals)

        if min_theta == np.inf:
            return "Функция неограничена сверху"

        leaving_var_idx = np.argmin(theta_vals)  # какой товар убираем из базиса
        leaving_var = basis[leaving_var_idx]

        # обновляем базис
        basis[leaving_var_idx] = entering_var

        # обнова текущего решения/плана
        b[entering_var] = min_theta
        for i in range(rows):
            if i != leaving_var_idx:
                if i != leaving_var:
                    b[basis[i]] -= min_theta * direction[i]
                else:
                    b[basis[i]] = 0
        b[leaving_var] = 0


c = np.array([1, 1, 0, 0, 0])
A = np.array([[-1, 1, 1, 0, 0],
              [1, 0, 0, 1, 0],
              [0, 1, 0, 0, 1]])
b = np.array([0, 0, 1, 3, 2]) # x
basis = np.array([2, 3, 4])

solution = simplex(c, A, b, basis)
print("Оптимальный план:", solution)