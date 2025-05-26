# from .sympy_pytential import sympy_pytential
import warnings
import sympy as sp
from sympy import symbols

# def sum_sympy_pytentials(pytentials):
#     """
#     sums a set of sympy pytentials into a single one in all variables

#     Sums a list of pytentials into a single one, renaming variables to identify
#     partioned quantities.

#     Args:
#         pytentials: a list of pytentials
#     """
   
#     fcns = [p.fcn_sym for p in pytentials]
#     return sympy_pytential(sum(fcns))

# def rename_sympy_variables(pytential, variables_to_rename):
#     """
#     renames variables in a sympy pytential according to the dictionary var_swap

#     Args:
#         variables_to_swap: a dictionary of variable substitutions
#     """
#     return sympy_pytential(pytential.fcn_sym.subs(variables_to_rename))

# def append_to_sympy_variables(pytential, suffix, variables_to_append=None):
#     """
#     Renames variables in a sympy pytential by appending a suffix

#     Args:
#         variables_to_append: a dictionary of variable to append the suffix to
#     """

#     def warn_if_not_true(condition, message):
#         if not condition:
#             warnings.warn(message, UserWarning)

#     if variables_to_append is None:
#         variables_to_append = pytential.vars
#     else:
#         checked_variables = all(v in pytential.vars for v in variables_to_append)
#         warn_if_not_true(checked_variables, "Not all variables are in the pytential")
    

#     appended_variables = symbols(' '.join([str(v)+suffix for v in variables_to_append]))
#     variables_to_rename = dict(zip(variables_to_append, appended_variables))
#     return rename_sympy_variables(pytential, variables_to_rename)

def expand_and_replace_variable_log_variable(expr):
    """
    Expands a SymPy expression and then replaces occurrences of
    variable * log(variable) with sp.xlogx(variable), where 'variable'
    must be a SymPy Symbol.

    The expansion includes:
    1. General algebraic expansion (e.g., distributing products over sums).
    2. Logarithmic expansion (e.g., log(a**b) -> b*log(a)).

    Args:
        expr: A SymPy expression.

    Returns:
        A SymPy expression that has been expanded and had matching patterns replaced.
    """
    # Step 1: Perform general algebraic expansion (e.g., for products, powers)
    # This helps in cases like x*(1+log(x)) -> x + x*log(x)
    expanded_expr = sp.expand(expr)
    
    # Step 2: Perform logarithmic expansion
    # force=True allows expansions like log(x**y) -> y*log(x) even if assumptions
    # on x and y are not explicitly set. This is applied to the already
    # generally expanded expression.
    # This helps in cases like log(x**x) -> x*log(x)
    fully_expanded_expr = sp.expand_log(expanded_expr, force=True)
        
    # Define a Wild symbol '_v_sym' that matches any SymPy Symbol.
    v_sym = sp.Wild('_v_sym', instanceof=sp.Symbol) 
    
    # Pattern 1: variable * log(variable)
    pattern1 = v_sym * sp.log(v_sym)
    
    # Pattern 2: log(variable) * variable (for completeness)
    pattern2 = sp.log(v_sym) * v_sym
    
    # Perform the replacements on the fully expanded expression.
    replaced_expr = fully_expanded_expr.replace(pattern1, sp.xlogx(v_sym))
    replaced_expr = replaced_expr.replace(pattern2, sp.xlogx(v_sym))
    
    return replaced_expr