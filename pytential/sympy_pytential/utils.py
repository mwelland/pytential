from sympy import symbols
from ..utils import find_matching_vars

# Should this be a pytential method instead of sympy? e.g.: Add a constraint to a general object?
# Should all constraints be separate pytentials...?

def get_sum_constraint_expressions(vars, pattern_var_pairs):
    """
    Sums all terms in the expression that match a pattern, subtracts the variable to collect

    Args:
        pattern_var_pairs (list): A list of tuples, where each tuple contains a pattern and the corresponding variable to collect.

    Returns:
        list: A list of summed expressions for each pattern and variable pair.
    """

    def _sum(vars_to_sum, var_to_collect):
        return sum(symbols(vars_to_sum)) - symbols(var_to_collect)

    return [_sum(find_matching_vars(vars, pattern), var_to_collect) for pattern, var_to_collect in pattern_var_pairs]