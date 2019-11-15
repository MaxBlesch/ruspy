"""
This module contains the main function to manage the simulation process. To simulate
a decision process in the model of John Rust's 1987 paper, it is sufficient to import
the function from this module and feed it with a init dictionary containing the
relevant variables.
"""
import numpy as np
import pandas as pd

from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.simulation.simulation_auxiliary import get_unobs_data
from ruspy.simulation.simulation_auxiliary import simulate_strategy


def simulate(init_dict, ev_known, trans_mat, shock=None):
    """
    The main function to simulate a decision process in the theoretical framework of
    John Rust's 1987 paper. It reads the inputs from the initiation dictionary and
    draws the random variables. It then calls the main subfunction with all the
    relevant parameters. So far, the feature of a agent's misbelief on the underlying
    transition probabilities is not implemented.

    :param init_dict: A dictionary containing the following variables as keys:

        :seed: (Digits)      : The seed determines random draws.
        :buses: (int)        : Number of buses to be simulated.
        :beta: (float)       : Discount factor.
        :periods: (int)      : Number of periods to be simulated.
        :params:             : A list or array of the cost parameters shaping the cost

    :param ev_known         : A 1d array containing the agent's expectation of the
                              value function in each state of dimension (num_states)
    :param trans_mat        : The transition matrix governing the discrete space Markov
                             decision process of dimension (num_states, num_states).
    :param shock            : A tuple of pandas.Series, where each Series name is
                             the scipy distribution function and the data is the loc
                             and scale specification.

    :return: The function returns the following objects:

        :df:         : A pandas dataframe containing for each observation the period,
                       state, decision and a Bus ID.
        :unobs:      : A three dimensional numpy array containing for each bus,
                       for each period random drawn utility for the decision to
                       maintain or replace the bus engine.
        :utilities:  : A two dimensional numpy array containing for each bus in each
                       period the utility as a float.
    """
    if "seed" in init_dict.keys():
        seed = init_dict["seed"]
    else:
        seed = np.random.randint(1, 100000)
    num_buses = init_dict["buses"]
    beta = init_dict["beta"]
    num_periods = init_dict["periods"]
    params = np.array(init_dict["params"])
    maint_func = lin_cost  # For now just set this to a linear cost function
    if ev_known.shape[0] != trans_mat.shape[0]:
        raise ValueError(
            "The transition matrix and the expected value of the agent "
            "need to have the same size."
        )
    num_states = ev_known.shape[0]
    costs = calc_obs_costs(num_states, maint_func, params)
    maint_shock_dist_name, repl_shock_dist_name, loc_scale = get_unobs_data(shock)
    states, decisions, utilities = simulate_strategy(
        num_periods,
        num_buses,
        costs,
        ev_known,
        trans_mat,
        beta,
        maint_shock_dist_name,
        repl_shock_dist_name,
        loc_scale,
        seed,
    )

    df = pd.DataFrame(
        {
            "Bus_ID": np.arange(1, num_buses + 1, dtype=np.uint32)
            .reshape(1, num_buses)
            .repeat(num_periods, axis=1)
            .flatten(),
            "period": np.arange(num_periods, dtype=np.uint32).repeat(num_buses),
            "state": states.flatten(),
            "decision": decisions.astype(np.uint8).flatten(),
            "utilities": utilities.flatten(),
        }
    )

    return df
