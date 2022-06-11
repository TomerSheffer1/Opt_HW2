import unittest
from Opt_HW2.src.constrained_min import *
from Opt_HW2.test.examples import *


class TestMinMethods(unittest.TestCase):
    def test_qp(self):
        m = 3
        step_len = 'wolfe'
        obj_tol = 1e-12
        param_tol = 1e-8
        barrier_tol = 1e-8
        max_iter = 1000
        c1 = 0.01
        rho = 0.5
        mu = 10
        t = 1
        x0 = np.array([0.1, 0.2, 0.7], dtype=np.float64)

        func = f1_const_min
        barrier_func = f1_phi
        eq_constraints_mat = np.array([1, 1, 1], dtype=np.float64)

        _, g, _ = func(x0)
        _, g_phi, _ = barrier_func(x0)
        g_tot = t*g + g_phi
        eq_constraints_rhs = np.vstack([-g_tot, 0])

        x_history_outer, f_history_outer, nu_hat, x_history, f_history = interior_pt(func, barrier_func, m, eq_constraints_mat, eq_constraints_rhs, x0, mu, t, step_len, max_iter, barrier_tol, obj_tol, param_tol, rho, c1)
        f1_verify_constraints(x_history[-1])
        plot_path(x_history, 1)

    def test_lp(self):
        m = 2
        step_len = 'wolfe'
        obj_tol = 1e-12
        param_tol = 1e-8
        barrier_tol = 1e-8
        max_iter = 1000
        c1 = 0.01
        rho = 0.5
        mu = 10
        t = 1
        x0 = np.array([0.5, 0.75], dtype=np.float64)

        func = f2_const_min
        barrier_func = f2_phi
        eq_constraints_mat = None

        _, g, _ = func(x0)
        _, g_phi, _ = barrier_func(x0)
        g_tot = t * g + g_phi
        eq_constraints_rhs = -g_tot

        x_history_outer, f_history_outer, nu_hat, x_history, f_history = interior_pt(func, barrier_func, m, eq_constraints_mat, eq_constraints_rhs, x0, mu, t,step_len, max_iter, barrier_tol, obj_tol, param_tol, rho, c1)
        f2_verify_constraints(x_history[-1])
        plot_path(x_history,2)

if __name__ == '__main__':
    unittest.main()
