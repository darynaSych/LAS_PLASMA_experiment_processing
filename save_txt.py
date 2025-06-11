import numpy as np

def save_arrays_to_txt(filename, arrays, column_names=None):
    """
    Save a list of arrays to a .txt file with optional column headers.

    Parameters:
    - filename: output .txt file
    - arrays: list of 1D numpy arrays of equal length
    - column_names: list of column names (same length as arrays)
    """
    # Stack arrays into columns
    data = np.column_stack(arrays)

    # Create header string
    if column_names is None:
        column_names = [f"col{i+1}" for i in range(len(arrays))]
    header = "\t".join(column_names)

    # Save to file
    np.savetxt(filename, data, header=header, comments='', fmt="%.18e")


def save_arrays_to_txt(filepath, arrays, column_names):
    """
    Save multiple 1D arrays into a .txt file with column headers for easy import into Origin.
    """
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("All arrays must have the same length.")

    data = np.column_stack(arrays)
    header = "# " + "\t".join(column_names)  # prepend '#' for Origin

    np.savetxt(filepath, data, header=header, comments='', fmt="%.18e")
