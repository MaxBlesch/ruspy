import numba
import numpy as np
from robupy.get_worst_case import get_worst_case_probs

@numba.jit(nopython=True)
def create_worst_trans_mat(trans_mat, v, rho):
    num_states = trans_mat.shape[0]
    worst_trans_mat = np.zeros(shape=(num_states, num_states), dtype=np.float64)
    for s in range(num_states):
        ind_non_zero = np.nonzero(trans_mat[s, :])[0]
        p_min = np.amin(ind_non_zero)
        p_max = np.amax(ind_non_zero)
        p = trans_mat[s, p_min: p_max + 1]
        v_intern = v[p_min: p_max + 1]
        worst_trans_mat[s, p_min: p_max + 1] = get_worst_case_probs(
            v_intern, p, rho, is_cost=False
        )
    return worst_trans_mat


@numba.jit(nopython=True)
def value_function_contraction(
    trans_mat, obs_costs, disc_fac, threshold=1e-12, max_it=1000000
):
    success = True
    num_eval = 0
    v_new = np.zeros(trans_mat.shape[0])

    converge_crit = threshold + 1

    while converge_crit > threshold:

        v = v_new

        ev = np.dot(trans_mat, v)

        maint_value = disc_fac * ev - obs_costs[:, 0]
        repl_value = (
            disc_fac * ev[0] - obs_costs[0, 1] - obs_costs[0, 0]
        )

        value_max = maint_value[0]

        v_new = value_max + np.log(
            np.exp(maint_value - value_max) + np.exp(repl_value - value_max)
        )

        converge_crit = np.max(np.abs(v_new - v))

        num_eval += 1

        if num_eval > max_it:
            success = False
            break

    return v_new, success, converge_crit, num_eval


@numba.jit(nopython=True)
def worst_value_fixp(
    v_start, trans_mat, obs_costs, disc_fac, rho, threshold=1e-8, max_it=10000
):
    worst_trans_mat = create_worst_trans_mat(trans_mat, v_start, rho)
    v_new, _, _, _ = value_function_contraction(
        worst_trans_mat, obs_costs, disc_fac, threshold, max_it)

    converge_crit_ev = threshold + 1
    num_eval = 0
    success = True

    while converge_crit_ev > threshold:
        v = v_new

        worst_trans_mat = create_worst_trans_mat(trans_mat, v, rho)

        ev = np.dot(worst_trans_mat, v)

        maint_value = disc_fac * ev - obs_costs[:, 0]
        repl_value = (
            disc_fac * ev[0] - obs_costs[0, 1] - obs_costs[0, 0]
        )

        # Select the minimal absolute value to rescale the value vector for the
        # exponential function.
        value_max = maint_value[0]

        v_new = value_max + np.log(
            np.exp(maint_value - value_max) + np.exp(repl_value - value_max)
        )

        converge_crit_ev = np.max(np.abs(v_new - v))

        num_eval += 1

        if num_eval > max_it:
            success = False
            break

    return v_new, worst_trans_mat, success, converge_crit_ev, num_eval





# @numba.jit(nopython=True)
# def calc_ev_fixp_worst(
#     ev_start, trans_mat, costs, disc_fac, rho, threshold=1e-8, max_it=10000
# ):
#     converge_crit = threshold + 1
#     ev_new = ev_start
#     num_eval = 0
#     success = True
#     while converge_crit > threshold:
#         ev = ev_new
#         maint_value = disc_fac * ev - costs[:, 0]
#         repl_value = disc_fac * ev[0] - costs[0, 1] - costs[0, 0]
#
#         # Select the minimal absolute value to rescale the value vector for the
#         # exponential function.
#         ev_min = maint_value[0]
#
#         log_sum = ev_min + np.log(
#             np.exp(maint_value - ev_min) + np.exp(repl_value - ev_min)
#         )
#         worst_trans_mat = create_worst_trans_mat(trans_mat, log_sum, rho)
#         ev_new = np.dot(worst_trans_mat, log_sum)
#
#         converge_crit = np.max(np.abs(ev_new - ev))
#         num_eval += 1
#
#         if num_eval > max_it:
#             success = False
#             break
#     return ev_new, success, converge_crit, num_eval


# @numba.guvectorize(
#     ["f8[:], f8[:], f8, f8[:]"], "(n_states), (n_states), () -> (n_states)",
#     nopython=True, target="cpu"
# )
# def create_worst_trans_mat_guvector(trans_row, v, rho, worst_trans_mat):
#     worst_trans_row = np.zeros(len(trans_row), dtype=np.float64)
#     ind_non_zero = np.nonzero(trans_row)[0]
#     p_min = np.amin(ind_non_zero)
#     p_max = np.amax(ind_non_zero)
#     p = trans_row[p_min : p_max + 1]
#     v_intern = v[p_min : p_max + 1]
#     worst_trans_row[p_min: p_max + 1] = get_worst_case_probs(
#         v_intern, p, rho, is_cost=False
#         )
#     worst_trans_mat[:] = worst_trans_row