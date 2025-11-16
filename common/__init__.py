# type: ignore
# Import order matters to avoid circular imports
# Import widgets after utility to avoid circular imports
import os

from . import resources
from .configuration import setup_config_store
from .utility import *  # noqa
from .utils.color_utility import *  # noqa


async def setup_config() -> None:
    """Sets up the config module."""
    filename: str = os.path.join(os.path.dirname(resources.__file__), "config.yml")
    await setup_config_store(filename=filename)
