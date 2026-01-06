from functools import wraps
import os

def with_identifier(func):
    @wraps(func)
    def wrapper(*args, identifier=None, **kwargs):
        if identifier:
            if "file_name" in kwargs:
                file_name, extension = os.path.splitext(kwargs["file_name"])
                kwargs["file_name"] = f"{identifier}_{file_name}{extension}"

            if "save_directory" in kwargs:
                kwargs["save_directory"] = os.path.join(kwargs["save_directory"], identifier)
        return func(*args, **kwargs)
    return wrapper

class DecoratedUtils:
    def __init__(self, function):
        self.original = function
        self.with_identifier = with_identifier(function)