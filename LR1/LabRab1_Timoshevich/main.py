import numpy as np


def solution(A, A_inv, x, i):
    n = len(A)
    if i < 0 or i >= n:
        print(f"Ошибка: индекс i = {i} выходит за пределы допустимого диапазона (0-{n - 1}).")
        return False, None

    l = np.dot(A_inv, x)  # шаг 1

    if l[i] == 0:
        return False, None

    l_e = l.copy()
    l_e[i] = -1  # шаг 2
    l_b = -l_e / l[i]  # шаг 3

    Q = np.eye(n)
    Q[:, i] = l_b  # шаг 4
    A_new_inv = np.dot(Q, A_inv)  # шаг 5

    return True, A_new_inv


A = np.array([[1, 0, 1],
              [0, 0, 0],
              [1, -1, 0]])

A_inv = np.array([[1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

x = np.array([1, 0, 1])
i = 2 

invertible, A_new_inv = solution(A, A_inv, x, i)

if invertible:
    print("Матрица обратима.")
    print("Обратная матрица:")
    print(A_new_inv)
else:
    print("Матрица необратима или произошла ошибка.")
