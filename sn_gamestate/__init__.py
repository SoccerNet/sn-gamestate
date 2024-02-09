from importlib import metadata
try:
    __version__ = metadata.version('sn-gamestate')
except metadata.PackageNotFoundError:
    __version__ = None