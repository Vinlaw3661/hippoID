import os

def verbose_print(text: str , verbose: bool = True) -> None:
    print(text) if verbose else None

def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)