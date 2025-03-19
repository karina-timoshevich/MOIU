from collections import Counter
import numpy as np


def balance(a, b, c):
    total_a: int = np.sum(a)
    total_b: int = np.sum(b)
    if total_a > total_b:
        b = np.hstack((b, np.array([total_a - total_b])))
        c = np.hstack((c, np.zeros((c.shape[0], 1))))
        print('a > b\n')
    elif total_a < total_b:
        a = np.hstack((a, np.array([total_b - total_a])))
        c = np.hstack((c, np.zeros((c.shape[0], 1))))
        print('a < b\n')
    else:
        print('a = b\n')
    return a, b, c


def check_optimality(X, B, c, m, n):
    A = np.zeros((m + n, m + n))
    b_vec = np.zeros(m + n)
    for b_i, (i, j) in enumerate(B):
        A[b_i, i] = 1
        A[b_i, m + j] = 1
        b_vec[b_i] = c[i, j]
    A[-1, 0] = 1
    print(f"A:\n{A}\n")
    print(f"b:\n{b_vec}\n")

    u_v = np.linalg.solve(A, b_vec)
    u = u_v[:m]
    v = u_v[m:]
    print(f'вектор u:\n{u}\n')
    print(f'вектор v:\n{v}\n')


    optimal = True
    for i in range(m):
        if optimal:
            for j in range(n):
                if u[i] + v[j] > c[i][j]:
                    optimal = False
                    B += [(i, j)]
                    print(f"[{i}] {u[i]} + [{j}] {v[j]} > [{i}][{j}] {c[i][j]} ")
                    print('план не оптимален\n')
                    print(f'вектор B:\n{B}\n')
                    break
    return optimal, B


def find_cycle(B):
    B_copy = B.copy()
    while True:
        i_counter = Counter([i for (i, _) in B_copy])
        j_counter = Counter([j for (_, j) in B_copy])
        i_to_rm = [i for i in i_counter if i_counter[i] == 1]
        j_to_rm = [j for j in j_counter if j_counter[j] == 1]
        if not i_to_rm and not j_to_rm:
            break

        B_copy = [(i, j) for (i, j) in B_copy if i not in i_to_rm
                  and j not in j_to_rm]
    print(f'новый вектор B:\n{B_copy}\n')
    plus, minus = [], []
    plus += [B_copy.pop()]

    while len(B_copy):
        if len(plus) > len(minus):
            for index, (i, j) in enumerate(B_copy):
                if plus[-1][0] == i or plus[-1][1] == j:
                    minus += [B_copy.pop(index)]
                    break
        else:
            for index, (i, j) in enumerate(B_copy):
                if minus[-1][0] == i or minus[-1][1] == j:
                    plus += [B_copy.pop(index)]
                    break
    print(f'+++++:\n{plus}\n')
    print(f'-----:\n{minus}\n')
    return plus, minus


def redistribute_resources(X, B, plus, minus):
    theta = min(X[i][j] for i, j in minus)
    print(f'значение тетта:\n{theta}\n')
    for i, j in plus:
        X[i][j] += theta
    for i, j in minus:
        X[i][j] -= theta
    for i, j in minus:
        if X[i][j] == 0:
            B.remove((i, j))
            break


def solve_transportation_problem(a, b, c):
    a, b, c = balance(a, b, c)
    print(f'новый вектор a:\n{a}\n')
    print(f'новый вектор b:\n{b}\n')
    print(f'новый вектор c:\n{c}\n')
    m, n = c.shape
    X = np.zeros_like(c)
    print(f'матрица X:\n{X}\n')
    B = []
    print(f'вектор B:\n{B}\n')
    print("1 фаза\n")
    i, j = 0, 0
    while i < m and j < n:
        minimum = min([a[i], b[j]])
        X[i, j] = minimum
        B += [(i, j)]
        a[i] -= minimum
        b[j] -= minimum
        if a[i] == 0:
            i += 1
        elif b[j] == 0:
            j += 1
        print(f'матрица X:\n{X}\n')
        print(f'вектор B:\n{B}\n')

    iter = 1
    print("2 фааза\n")
    while True:
        print(f" {iter} итерация\n")
        print(f'матрица X:\n{X}\n')
        print(f'вектор B:\n{B}\n')
        optimal, B = check_optimality(X, B, c, m, n)
        if optimal:
            print('план оптимален\n')
            return X

        plus, minus = find_cycle(B)
        redistribute_resources(X, B, plus, minus)
        print(f'матрица X:\n{X}\n')
        print(f'матрица B:\n{B}\n')
        iter += 1


if __name__ == "__main__":
    a = np.array([100, 300, 300])
    b = np.array([300, 200, 200])
    c = np.array([[8, 4, 1],
                  [8, 4, 3],
                  [9, 7, 5]])
    print(f'вектор a:\n{a}\n')
    print(f'вектор b:\n{b}\n')
    print(f'вектор c:\n{c}\n')
    optimal = solve_transportation_problem(a, b, c)
    print(f"оптимальная матрица X: \n{optimal}")