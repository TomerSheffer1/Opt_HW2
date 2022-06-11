from Opt_HW2.src.utils import *


def interior_pt(func, barrier_func, m, eq_constraints_mat, eq_constraints_rhs, x0, mu, t, step_len, max_iter, barrier_tol, obj_tol, param_tol, rho, c1):
    x = x0
    f_x, g_x, h_x = func(x)
    f_x_phi, g_x_phi, h_x_phi = barrier_func(x)
    x_history = []
    f_history = []
    x_history_outer = []
    f_history_outer = []
    nu_hat = []

    f_x = t * f_x + f_x_phi
    g_x = t * g_x + g_x_phi
    h_x = t * h_x + h_x_phi
    print(f"Iteration 0: x0 = {x}, f(x0) = {f_x}")

    l = 0
    while l < max_iter:
        '''
        Determine p_nt (newton direction) and w (dual problem)
        Solve the linear system:

        [Hessian(f)   A^T] * [  p_nt ] = [-Grad(f)]
        [    A         0 ]   [   w   ]   [    0   ]

        '''
        if eq_constraints_mat is not None:
            A = eq_constraints_mat.reshape((1, m))
            b1 = np.column_stack((h_x, A.T))
            b2 = np.column_stack((A, 0))
            B = np.row_stack((b1, b2))
            eq_constraints_rhs = np.vstack([-g_x, 0])
        else:
            B = h_x
            eq_constraints_rhs = -g_x

        # update run history
        x_prev = x
        f_prev = f_x
        x_history.append(x0)
        f_history.append(f_x)
        x_history_outer.append(x0)
        f_history_outer.append(f_x)

        k = 0
        while k < max_iter:
            # validate convergence - step tolerance
            if k != 0 and sum(abs(x - x_prev)) < param_tol:
                break

            # solve linear system
            p = np.linalg.solve(B, eq_constraints_rhs)[: m]
            p = np.squeeze(p)

            if k != 0 and (f_prev - f_x < obj_tol):
                break

            # validate wolfe conditions and update step length
            if step_len == "wolfe":
                alpha = wolf_update(func, p, x, c1, rho)
            else:
                alpha = step_len

            x_prev = x
            f_prev = f_x

            # update next step
            x = x + alpha * p
            f_x, g_x, h_x = func(x)
            f_x_phi, g_x_phi, h_x_phi = barrier_func(x)

            print(f"Iteration {k + 1}: x{k + 1} = {x}, f(x{k + 1}) = {f_x}")

            # calc value, gradient and hessian of the modified function
            f_x = t * f_x + f_x_phi
            g_x = t * g_x + g_x_phi
            h_x = t * h_x + h_x_phi

            # update run history
            x_history.append(x)
            f_history.append(f_x)

            k += 1

        # update nu_hat
        x_history_outer.append(x)
        f_history_outer.append(f_x)
        nu_hat.append(p[-1] / t)

        # validate convergence - barrier tolerance
        if m / t < barrier_tol:
            return x_history_outer, f_history_outer, nu_hat, x_history, f_history

        # update run params
        t = mu * t
        l += 1

    return x_history_outer, f_history_outer, nu_hat, x_history, f_history


def wolf_update(f, p, x, c, rho):
    alpha = 1
    while f(x + alpha * p)[0] > f(x)[0] + c * alpha * np.matmul(f(x)[1].transpose(), p)[0]:
        alpha = alpha * rho
    return alpha
