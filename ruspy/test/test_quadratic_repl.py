import pickle as pkl

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import quadratic_costs
from ruspy.model_code.cost_functions import quadratic_costs_dev
from ruspy.ruspy_config import TEST_RESOURCES_DIR


TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    beta = 0.9999
    num_states = 90
    scale = 0.001
    init_dict = {
        "groups": "group_4",
        "binsize": 5000,
        "model_specifications": {
            "discount_factor": beta,
            "number_states": num_states,
            "maint_cost_func": "quadratic",
            "cost_scale": scale,
        },
        "optimizer": {
            "optimizer_name": "BFGS",
            "use_gradient": "yes",
            "use_search_bounds": "yes",
        },
    }
    df = pkl.load(open(TEST_FOLDER + "group_4.pkl", "rb"))
    result_trans, result_fixp = estimate(init_dict, df)
    out["trans_est"] = result_trans["x"]
    out["params_est"] = result_fixp["x"]
    out["trans_ll"] = result_trans["fun"]
    out["cost_ll"] = result_fixp["fun"]
    out["states"] = df.loc[(slice(None), slice(1, None)), "state"].to_numpy()
    out["decisions"] = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy()
    out["beta"] = beta
    out["num_states"] = num_states
    out["scale"] = scale
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.402
    return out


def test_repl_trans(inputs, outputs):
    assert_array_almost_equal(inputs["trans_est"], outputs["trans_base"])


def test_trans_ll(inputs, outputs):
    assert_allclose(inputs["trans_ll"], outputs["trans_ll"])


def test_cost_ll(inputs, outputs):
    # This is as precise as the paper gets
    assert_allclose(np.round(inputs["cost_ll"], 3), outputs["cost_ll"])


def test_ll_params_derivative(inputs, outputs):
    num_states = inputs["num_states"]
    trans_mat = create_transition_matrix(num_states, outputs["trans_base"])
    state_mat = create_state_matrix(inputs["states"], num_states)
    endog = inputs["decisions"]
    decision_mat = np.vstack(((1 - endog), endog))
    beta = inputs["beta"]
    assert_array_almost_equal(
        derivative_loglike_cost_params(
            inputs["params_est"],
            quadratic_costs,
            quadratic_costs_dev,
            num_states,
            trans_mat,
            state_mat,
            decision_mat,
            beta,
            inputs["scale"],
        ),
        np.array([0, 0, 0]),
        decimal=2,
    )
