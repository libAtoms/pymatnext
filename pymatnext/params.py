from copy import deepcopy

class ParamError(ValueError):
    pass

def check_fill_defaults(params_section, defaults_section, label="top", verbose=False):
    """check that a toml parsed dict is consistent with the template, and
    fill in whatever defaults are missing

    Parameters
    ----------
    param_section: dict
        toml parsed parameters dict
    defaults_section: dict
        corresponding template dict
    label: str, default "top"
        string to preface verbose output lines with
    verbose: bool, default False
        produce verbose (debugging) output
    """
    if len(label) != 0:
        label = label + " "

    if type(params_section) != type(defaults_section):
        raise ParamError(f"{label}: Params are type {type(params_section)}, should be {type(defaults_section)}")

    if verbose:
        print("check_fill_defaults", label, "START")
        # print("check_fill_defaults", label, "params")
        # print(params_section)
        # print("check_fill_defaults", label, "defaults")
        # print(defaults_section)

    if isinstance(defaults_section, dict):
        # compare dicts
        if verbose:
            print("check_fill_defaults", label, "is dict")

        ignore_unknown_keys = False
        for k in defaults_section:
            if verbose:
                print("check_fill_defaults", label, "check key", k)
            if k == "_IGNORE_":
                if verbose:
                    print("check_fill_defaults", label, "ignoring")
                ignore_unknown_keys = True
                continue

            req_value = None
            try:
                req = len(defaults_section[k]) > 0 and defaults_section[k][0] == "_REQ_"
                if req:
                    if len(defaults_section[k]) != 2:
                        raise ParamError(f"{label}: Key {k} got defaults _REQ_ but item has length {len(defaults_section[k])} != 2 {defaults_section[k]}")
                    req_value = defaults_section[k][1]
            except (TypeError, KeyError):
                req = False

            if req_value == "_IGNORE_" or (req_value is None and defaults_section[k] == "_IGNORE_"):
                # ignore contents this key
                if req and k not in params_section:
                    # it's required but not present
                    raise ParamError(f"{label}: Required key {k} is missing")

                # ignoring, skip any further check
                continue

            if k not in params_section:
                # value is missing
                # check for required
                if req:
                    raise ParamError(f"{label}: Required key {k} is missing")
                # copy in from defaults
                params_section[k] = deepcopy(defaults_section[k])
                if isinstance(params_section[k], dict):
                    params_section[k].pop("_IGNORE_", None)
            else:
                # key is present
                if req:
                    if verbose:
                        print("check_fill_defaults", label, "check contained required dict")
                    # check type against default example
                    check_fill_defaults(params_section[k], req_value, label=f"{label} / {k}", verbose=verbose)
                else:
                    if verbose:
                        print("check_fill_defaults", label, "check contained optional dict")
                    # check type against default
                    check_fill_defaults(params_section[k], defaults_section[k], label=f"{label} / {k}", verbose=verbose)

        if not ignore_unknown_keys:
            unknown_keys = set(params_section) - set(defaults_section)
            if len(unknown_keys) > 0:
                raise ParamError(f"{label}: Unknown keys {unknown_keys}")
    elif isinstance(defaults_section, (list, tuple)):
        # NOTE: do we need support for variable length lists, or lists with free types?
        if len(params_section) != len(defaults_section):
            raise ParamError(f"{label}: Params has length {len(params_section)} != {len(defaults_section)}")
        for item_i, (param_item, defaults_item) in enumerate(zip(params_section, defaults_section)):
            check_fill_defaults(param_item, defaults_item, label=f"{label} / i={item_i}", verbose=verbose)
