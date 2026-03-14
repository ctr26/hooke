def dict_to_ordered_tuple(d: dict, keys: list[str]) -> tuple:
    """Very simple utility function to convert a dictionary to an ordered tuple."""
    return tuple(d[key] for key in keys)
