"""Utility functions.
"""

import re
from collections.abc import Callable
from pathlib import Path
from textwrap import indent


# why is this a thing? read a file and return a list?
def load_tf_names(tf_file: Path) -> list[str]:
    """
    Parameters
    ----------
    tf_file : Path
        The path to the file listing transcription factors

    Returns
    -------
    list[str] :
        A list of transcription factor names
    """

    with open(tf_file) as file:
        tfs_in_file = [line.strip() for line in file.readlines()]

    return tfs_in_file


# taken from `scanpy <https://github.com/scverse/scanpy/>`_
def _doc_params[T: Callable | type](**replacements: str) -> Callable[[T], T]:
    def dec(obj: T) -> T:
        _leading_whitespace_re = re.compile("(^[ ]*)(?:[^ \n])", re.MULTILINE)

        assert obj.__doc__
        assert "\t" not in obj.__doc__

        # The first line of the docstring is unindented,
        # so find indent size starting after it.
        start_line_2 = obj.__doc__.find("\n") + 1
        assert start_line_2 > 0, f"{obj.__name__} has single-line docstring."
        n_spaces = min(
            len(m.group(1))
            for m in _leading_whitespace_re.finditer(obj.__doc__[start_line_2:])
        )

        # The placeholder is already indented, so only indent subsequent lines
        indented_replacements = {
            k: indent(v, " " * n_spaces)[n_spaces:] for k, v in replacements.items()
        }
        obj.__doc__ = obj.__doc__.format_map(indented_replacements)
        return obj

    return dec