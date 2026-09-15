import importlib

from .loop_exit_params import NSLoopExitParams

class NSLoopExit:
    """Implement exit conditions for main NS loop

    Parameters
    ----------
    params: NSLoopExitParams
        parameters defining exit conditions
    ns: NS
        nested sampling object
    """

    def __init__(self, params, ns):
        params = NSLoopExitParams.model_validate(params)

        exit_module = params.module
        if exit_module is not None and exit_module != "_NONE_":
            self.exit_evaluator = importlib.import_module(exit_module).ExitLoop(ns=ns, **params.module_kwargs)
        else:
            self.exit_evaluator = lambda loop_iter, max_val: False

    def __call__(self, ns, loop_iter):
        return self.exit_evaluator(loop_iter, ns.max_val)
