import os
from typing import Callable 

def do_not_print(text: str) -> None:
    return 

def verbose_print(verbose: bool = False) -> Callable[[str], None]:
    if verbose:
        return print
    return do_not_print

def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)