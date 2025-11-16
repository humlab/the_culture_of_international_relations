# type: ignore
# Import order matters to avoid circular imports
# Import widgets after utility to avoid circular imports
import os

from . import resources
from .color_utility import *  # noqa
from .configuration import setup_config_store
# from .file_utility import FileUtility  # noqa
from .utility import *  # noqa
from .utility import plot_wordcloud  # noqa


async def setup_config() -> None:
    """Sets up the config module."""
    filename: str = os.path.join(os.path.dirname(resources.__file__), "config.yml")
    await setup_config_store(filename=filename)
