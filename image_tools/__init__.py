from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("image-tools")
except PackageNotFoundError:
    # package is not installed
    pass

from . import similarity_transforms, registration, io
