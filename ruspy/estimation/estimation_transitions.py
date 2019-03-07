"""
This module contains the function necessary for the estimation process of transition
probabilities.
"""
import numpy as np
from math import log
import scipy.optimize as opt
import numba


def estimate_transitions_5000(df):
    """
    The sub function for estimating the transition probabilities. This function
    manages the estimation process of the transition probaiblities and calls the
    necessary subfunctions.

    :param df: A pandas dataframe, which contains for each observation the Bus ID,
    the current state of the bus, the current period and the decision made in this
    period.
    :return: The optimization result of the transition probabilities estimation as a
    dictionary.
    """
    transition_count = [0]
    num_bus = len(df['Bus_ID'].unique())
    num_periods = int(df.shape[0] / num_bus)
    states = df['state'].values.reshape(num_bus, num_periods)
    decisions = df['decision'].values.reshape(num_bus, num_periods)
    transition_count = count_transitions_5000(transition_count, num_bus, num_periods,
                                              states, decisions)
    dim = len(transition_count)
    x_0 = np.full(dim, 0.1)
    result_transitions = opt.minimize(loglike, args=transition_count, x0=x_0,
                                      bounds=[(1e-6, 1)] * dim,
                                      constraints=({'type': 'eq', "fun": cond_sum_one}))
    return result_transitions


@numba.jit(nopython=True)
def count_transitions_5000(transition_count, num_bus, num_periods, states, decisions):
    """
    This function counts how often the buses increased their state by 0, by 1 etc.

    :param transition_count: A list with only one integer zero.
    :param num_bus:          The number of buses in the samples.
    :type num_bus:           int
    :param num_periods:      The number of periods the buses drove.
    :type num_periods:       int
    :param states:           A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
    :param decisions:        A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.

    :return: A list with the highest increase as maximal index and the increase
    counts as entries.
    """

    for bus in range(num_bus):
        for period in range(num_periods - 1):
            if decisions[bus, period] == 0:
                increase = states[bus, period + 1] - states[bus, period]
            else:
                increase = 1
            if increase >= len(transition_count):
                transition_count_new = [0] * (increase + 1)
                for i in range(len(transition_count)):
                    transition_count_new[i] = transition_count[i]
                transition_count = transition_count_new
            transition_count[increase] += 1
    return transition_count


def cond_sum_one(inputs):
    """A constraint which checks the sum of the transition probabilities.

    :param inputs: A numpy array of transition probabilities.
    :return: Should return 0.
    """
    total = 1 - np.sum(inputs)
    return total


def loglike(params, transition_list):
    """
    The loglikelihood function for estimating the transition probabilities.

    :param params:          An array of choice probabilities.
    :param transition_list: A list with the highest increase as maximal index and the
                            increase counts as entries.

    :return: The negative loglikelihood value for minimizing the second liklihood
    function.
    """
    ll = 0
    for i in range(len(params)):
        ll = ll + transition_list[i] * log(params[i])
    return -ll