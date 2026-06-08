__all__ = [
    "arms",
]

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("robot-arms")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except Exception:
    __version__ = "0.0.0"

