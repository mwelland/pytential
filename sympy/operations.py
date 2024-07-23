from . import sympy_potential

def sum_sympy_phases(phases, rename = True):
    """
    sums a set of sympy phases into a single potential in all variables

    Sums a list of potentials into a single one, renaming variables to identify
    partioned quantities.

    Args:
        phases: a list of potentials
    """
    # TODO: Currently relies on sympy reversion. Should be a summation of lambda functions.

    # if rename:
    #     if not phase_ids:
    #         phase_ids = [chr(i+97) for i in range(len(phases))]
    #     [p.rename_vars(pi) for p,pi in zip(phases, phase_ids)]
    
    fcns = [p.fcn_sym for p in phases]
    return sympy_potential(sum(fcns))