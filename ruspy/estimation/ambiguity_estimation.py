import numba
import numpy as np
from robupy.get_worst_case import get_worst_case_probs


@numba.jit(nopython=True)
def create_worst_trans_mat(trans_mat, v, rho):
    num_states = trans_mat.shape[0]
    worst_trans_mat = np.zeros(shape=(num_states, num_states), dtype=np.float64)
    for s in range(num_states):
        p = trans_mat[s, s: s + 3]
        v_intern = v[0: s + 3]
        worst_trans_mat[s, s: s + 3] = get_worst_case_probs(
            v_intern, p, rho, is_cost=False
        )
    return worst_trans_mat


@numba.jit(nopython=True)
def calc_fixp_worst(
    ev_start, trans_mat, costs, disc_fac, rho, threshold=1e-8, max_it=10000
):
    converge_crit = threshold + 1
    ev_new = ev_start
    num_eval = 0
    success = True
    while converge_crit > threshold:
        ev = ev_new
        maint_value = disc_fac * ev - costs[:, 0]
        repl_value = disc_fac * ev[0] - costs[0, 1] - costs[0, 0]

        # Select the minimal absolute value to rescale the value vector for the
        # exponential function.
        ev_min = maint_value[0]

        log_sum = ev_min + np.log(
            np.exp(maint_value - ev_min) + np.exp(repl_value - ev_min)
        )
        worst_trans_mat = create_worst_trans_mat(trans_mat, log_sum, rho)
        ev_new = np.dot(worst_trans_mat, log_sum)

        converge_crit = np.max(np.abs(ev_new - ev))
        num_eval += 1

        if num_eval > max_it:
            success = False
            break
    return ev_new, success, converge_crit, num_eval
