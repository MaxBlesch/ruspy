"""
This module contains tests for the data and estimation code of the ruspy project. The
settings for this tests is specified in resources/replication_test/init_replication.yml.
The test first reads the original data, then processes the data to a pandas DataFrame
suitable for the estimation process. After estimating all the relevant parameters,
they are compared to the results, from the paper. As this test runs the complete
data_reading, data processing and runs several times the NFXP it is the one with the
longest test time.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import lin_cost_dev


TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 90
    scale = 1e-3
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "linear",
            "cost_scale": scale,
        },
        "optimizer": {"approach": "NFXP", "algorithm": "scipy_L-BFGS-B",
                      "gradient": "No",
                      "params": pd.DataFrame(
                data=[10, 1, 0],
                columns=["value"],
                index=["RC", "theta_11", "omega"]
            ),
                      "constraints": [{"loc": "omega", "type": "fixed"}]
                      },
    }

    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")
    result_trans, result_fixp = estimate(init_dict, df)
    out["trans_est"] = result_trans["x"]
    out["params_est"] = result_fixp["x"]
    out["trans_ll"] = result_trans["fun"]
    out["cost_ll"] = result_fixp["fun"]
    out["states"] = df.loc[(slice(None), slice(1, None)), "state"].to_numpy(int)
    out["decisions"] = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy(int)
    out["disc_fac"] = disc_fac
    out["num_states"] = num_states
    out["scale"] = scale
    out["status"] = result_fixp["status"]
    out["params"] = init_dict["optimizer"]["params"]
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["params_base"] = np.append(np.loadtxt(TEST_FOLDER +
                                              "repl_params_linear.txt"), 0)
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.584
    return out


def test_repl_params(inputs, outputs):
    # This is as precise as the paper gets
    assert_array_almost_equal(inputs["params_est"], outputs["params_base"], decimal=2)


def test_repl_trans(inputs, outputs):
    assert_array_almost_equal(inputs["trans_est"], outputs["trans_base"])


def test_trans_ll(inputs, outputs):
    assert_allclose(inputs["trans_ll"], outputs["trans_ll"])


def test_cost_ll(inputs, outputs):
    # This is as precise as the paper gets
    assert_allclose(inputs["cost_ll"], outputs["cost_ll"], atol=1e-3)