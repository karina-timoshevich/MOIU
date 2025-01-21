import numpy as np


def is_invertible_and_find_inverse(A, A_inv, x, i):

    n = len(A)
    l = np.dot(A_inv, x)

    if l[i] == 0:
        return False, None

    l_e = l.copy()
    l_e[i] = -1
    l_b = -l_e / l[i]

    Q = np.eye(n)
    Q[:, i] = l_b
    A_new_inv = np.dot(Q, A_inv)

    return True, A_new_inv

A = np.array([[1, 0, 1],
              [0, 0, 0],
              [1, -1, 0]])

A_inv = np.array([[1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

x = np.array([1, 0, 1])
i = 2

invertible, A_new_inv = is_invertible_and_find_inverse(A, A_inv, x, i)

if invertible:
    print("Матрица обратима.")
    print("Обратная матрица:")
    print(A_new_inv)
else:
    print("Матрица необратима.")
