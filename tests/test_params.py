from copy import deepcopy

import pytest

from pymatnext.params import check_fill_defaults, ParamError


def test_fill_in_defaults():
    defaults = { "step_size": ["_REQ_", 1.0], "n_steps": [1, 2], "move_type" : { "pos": 5, "cell": 6 } }

    params = {"step_size": 2.0}
    check_fill_defaults(params, defaults)

    # make sure defaults were filled in properly

    # replace required examples with specific values
    defaults["step_size"] = 2.0
    assert params == defaults


def test_missing_req():
    defaults = { "step_size": ["_REQ_", 1.0], "n_steps": [1, 2], "move_type" : { "pos": 5, "cell": 6 } }

    with pytest.raises(ParamError):
        check_fill_defaults({"n_steps": [2, 3]}, defaults)


def test_mismatched_type_default():
    defaults = { "step_size": ["_REQ_", 1.0], "n_steps": [1, 2], "move_type" : { "pos": 5, "cell": 6 } }

    with pytest.raises(ParamError):
        check_fill_defaults({"step_size": True}, defaults)


def test_mismatched_list_type():
    defaults = { "step_size": ["_REQ_", 1.0], "n_steps": [1, 2], "move_type" : { "pos": 5, "cell": 6 } }

    with pytest.raises(ParamError):
        check_fill_defaults({"step_size": 2.0, "n_steps": ["1", "2"]}, defaults)


def test_mismatched_list_len():
    defaults = { "step_size": ["_REQ_", 1.0], "n_steps": [1, 2], "move_type" : { "pos": 5, "cell": 6 } }

    with pytest.raises(ParamError):
        check_fill_defaults({"step_size": 2.0, "n_steps": [1, 2, 3]}, defaults)


def test_ignore_item():
    # ignored section is present but check_fill_defaults doesn't check its content
    defaults = { "step_size": ["_REQ_", 1.0], "n_steps": [1, 2], "move_type" : "_IGNORE_" }
    params = { "step_size":  2.0, "n_steps": [2, 3], "move_type" : { "pos": 5, "cell": 6 } }

    params_orig = deepcopy(params)
    check_fill_defaults(params, defaults)
    assert params == params_orig

    # ignored section is required but missing
    defaults = { "step_size": ["_REQ_", 1.0], "n_steps": [1, 2], "move_type" : ["_REQ_", "_IGNORE_"] }
    params = { "step_size":  2.0, "n_steps": [2, 3] }

    with pytest.raises(ParamError):
        check_fill_defaults(params, defaults)
