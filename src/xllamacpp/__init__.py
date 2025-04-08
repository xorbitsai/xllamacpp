from .xllamacpp import *

from . import _version

__version__ = _version.get_versions()["version"]
if __version__ == "0+unknown":
    print(_version.get_versions())
