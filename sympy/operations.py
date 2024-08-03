from .sympy_pytential import sympy_pytential
import warnings
from sympy import symbols

def sum_sympy_pytentials(pytentials):
    """
    sums a set of sympy pytentials into a single one in all variables

    Sums a list of pytentials into a single one, renaming variables to identify
    partioned quantities.

    Args:
        pytentials: a list of pytentials
    """
   
    fcns = [p.fcn_sym for p in pytentials]
    return sympy_pytential(sum(fcns))

def rename_sympy_variables(pytential, variables_to_rename):
    """
    renames variables in a sympy pytential according to the dictionary var_swap

    Args:
        variables_to_swap: a dictionary of variable substitutions
    """
    return sympy_pytential(pytential.fcn_sym.subs(variables_to_rename))

def append_to_sympy_variables(pytential, suffix, variables_to_append=None):
    """
    Renames variables in a sympy pytential by appending a suffix

    Args:
        variables_to_append: a dictionary of variable to append the suffix to
    """

    def warn_if_not_true(condition, message):
        if not condition:
            warnings.warn(message, UserWarning)

    if variables_to_append is None:
        variables_to_append = pytential.vars
    else:
        checked_variables = all(v in pytential.vars for v in variables_to_append)
        warn_if_not_true(checked_variables, "Not all variables are in the pytential")
    

    appended_variables = symbols(' '.join([str(v)+suffix for v in variables_to_append]))
    variables_to_rename = dict(zip(variables_to_append, appended_variables))
    return rename_sympy_variables(pytential, variables_to_rename)