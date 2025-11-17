import os

import dotenv
from loguru import logger

from .provider import ConfigLike, ConfigStore

dotenv.load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))


async def setup_config_store(filename: str = "config.yml", force: bool = False) -> None:

    config_file: str = os.getenv("CONFIG_FILE", filename)
    store: ConfigStore = ConfigStore.get_instance()

    if store.is_configured() and not force:
        return

    store.configure_context(source=config_file, env_filename=".env", env_prefix="SEAD_AUTHORITY")

    assert store.is_configured(), "Config Store failed to configure properly"

    cfg: ConfigLike | None = store.config()
    if not cfg:
        raise ValueError("Config Store did not return a config")

    cfg.update({"runtime:config_file": config_file})

    logger.info("Config Store initialized successfully.")
