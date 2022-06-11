import numpy as np


def f1_const_min(x0):
    x, y, z = x0
    f = x**2 + y**2 + (z+1)**2
    g = np.array([2*x, 2*y, 2*z+2]).reshape(3,1)
    h = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    return f, g, h


def f1_phi(x0):
    x, y, z = x0
    f = - (np.log(x) + np.log(y) + np.log(z))
    g = np.array([- 1 / x, - 1 / y, - 1 / z]).reshape((3,1))
    # h = np.array([[1 / x ** 2, 0, 0], [0, 1 / y ** 2, 0], [0, 0, 1 / z ** 2]])
    h = np.zeros((3,3))
    for i in range(3):
        h[i,i] = (g[i,0]/(-x0[i])) ** 2

    return f, g, h


def f1_verify_constraints(x_final):
    x, y, z = x_final
    c1 = np.isclose(x+y+z, 1)
    c2 = (x >= 0)
    c3 = (y >= 0)
    c4 = (z >= 0)
    print(f"Constraint 1: x + y + z = 1 {c1}")
    print(f"Constraint 2: x >= 0 {c2}")
    print(f"Constraint 3: y >= 0 {c3}")
    print(f"Constraint 4: z >= 0 {c4}")


def f2_const_min(x0):
    x, y = x0
    f = - (x + y)
    g = np.array([-1, -1]).reshape(2,1)
    h = np.array([[0, 0], [0, 0]])

    return f, g, h


def f2_phi(x0):
    x, y = x0
    f = - (np.log(x+y-1) + np.log(1-y) + np.log(2-x) + np.log(y))

    g_1 = np.array([-1/(x+y-1), -1/(x+y-1)]).reshape((2,1))
    g_2 = np.array([0, 1/(1-y)]).reshape((2,1))
    g_3 = np.array([-1/(x-2), 0]).reshape((2,1))
    g_4 = np.array([0, -1/y]).reshape((2,1))
    g = g_1 + g_2 + g_3 + g_4

    h_1 = np.matmul(g_1, g_1.T) / (-x-y+1)**2
    h_2 = np.matmul(g_2, g_2.T) / (-1+y)**2
    h_3 = np.matmul(g_3, g_3.T) / (-2+x)**2
    h_4 = np.matmul(g_4, g_4.T) / (-y)**2
    h = h_1 + h_2 + h_3 + h_4

    return f, g, h


def f2_verify_constraints(x_final):
    x, y = x_final
    c1 = (y >= -x+1)
    c2 = (y <= 1)
    c3 = (x <= 2)
    c4 = (y >= 0)
    print(f"Constraint 1: y >= -x + 1 {c1}")
    print(f"Constraint 2: y <= 1 {c2}")
    print(f"Constraint 3: x <= 2 {c3}")
    print(f"Constraint 4: y >= 0 {c4}")
