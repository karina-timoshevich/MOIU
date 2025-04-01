import numpy as np


def get_basis_vector(c, B):
    i = 0
    c_b = [0 for _ in B]
    for index in B:
        c_b[i] = c[index]
        i += 1
    return c_b


def check(delta_x):
    for i in range(len(delta_x)):
        if delta_x[i] < 0:
            return i
    return -1

# f(x) = c'·x+1/2·x'·D·x -> min;
def main():
    c = np.array([-8, -6, -4, -6])
    A = np.array([[1, 0, 2, 1],
                  [0, 1, -1, 2]])
    x = np.array([2, 3, 0, 0])
    JB = np.array([0, 1])
    JBs = np.array([0, 1])  # расширенная опора ограничений
    D = np.array([[2, 1, 1, 0],
                  [1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 0]])
    counter = 0
    print("На старте по условию задачи имеем:")
    print("Матрица A: ", A, sep="\n")
    print("Матрица D: ", D, sep="\n")
    print("Вектор JB: ", JB, sep="")
    print("Вектор с: ", c, sep="")
    print("Вектор x: ", x, sep="")
    while True:
        counter += 1
        print(counter)
        print(f"=======================================================")
        A_b = A[:, JB]
        A_b_inv = np.linalg.inv(A_b)

        # ШАГ 1 (b) Вычислим векторы c(x); u(x) и ∆(x):
        c_x = c + np.dot(x, D)
        c_b = get_basis_vector(c_x, JB)
        c_b = [i * (-1) for i in c_b]
        u_x = np.dot(c_b, A_b_inv)
        delta_x = np.dot(u_x, A) + c_x
        print("Вектор с(x): ", c_x, sep="")
        print("Вектор u(x): ", u_x, sep="")
        print("Вектор ∆(x): ", delta_x, sep="")

        # ШАГ 2
        j0 = check(delta_x)
        if j0 == -1:
            print("Оптимальный план задачи: ", x, sep="")
            break
        print(
            "Емть отрицательные компоненты вектора ∆(x)"
        )  # сюда шаг 3 - индекс отрицательной компоненты в j0 из выше

        # ШАГ 4
        l = np.zeros(len(x))
        l[j0] = 1
        A_b_ext = A[:, JBs]

        # [D[JBs, :][:, JBs] подматрица D, сост из элем, стоящих на пересечении стр и стл
        H = np.bmat(
            [[D[JBs, :][:, JBs], A_b_ext.T],
             [A_b_ext, np.zeros((len(A), len(A)))]]
        )
        H_inv = np.array(np.linalg.inv(H))

        b_starred = np.concatenate((D[JBs, j0], A[:, j0]))
        x_temp = np.dot(-H_inv, b_starred)
        l[: len(JBs)] = x_temp[: len(JBs)]

        # Шаг 5
        sigma = np.dot(np.dot(l, D), l)
        theta = {}
        theta[j0] = np.inf if sigma == 0 else np.abs(delta_x[j0]) / sigma

        for j in JBs:
            if l[j] < 0:
                theta[j] = -x[j] / l[j]
            else:
                theta[j] = np.inf

        j_s = min(theta, key=theta.get)  # индекс минимума
        theta_0 = theta[j_s]  # минимальное

        if theta_0 == np.inf:
            print("целевая функция задачи не ограничена снизу на множестве допустимых планом")

        # ШАГ 6 обновляем допустимый план
        x = x + theta_0 * l
        # обновим теперь опору ограничений расширенную опору ограничений
        if j_s == j0:
            JBs = np.append(JBs, j_s)
        elif j_s in JBs and j_s not in JB:
            JBs = np.delete(JBs, j_s)
        elif j_s in JB:
            third_condition = False
            s = JB.index(j_s)  # индекс j* идёт s-м по счёту в Jb

            # oбновляем опоры ограничений
            for j_plus in set(JBs).difference(JB): # Jb*\Jb
                if (np.dot(A_b_inv, A[:, j_plus]))[s] != 0:
                    third_condition = True
                    JB[s] = j_plus
                    JBs = np.delete(JBs, j_s)

            if not third_condition:
                JB[JB.index(j_s)] = j0
                JBs[JBs.index(j_s)] = j0
            print("Обновленные опоры ограничений: ", JBs, sep="")


if __name__ == "__main__":
    main()
