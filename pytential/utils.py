from fnmatch import filter

def find_matching_vars(vars, pattern):
    """
    Finds variables in a list of variables that match a pattern.

    Args:
        vars (list): A list of variables.
        pattern (str): The pattern to match.

    Returns:
        list: A list of variables that match the pattern.
    """

    return filter(vars, pattern)