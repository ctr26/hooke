from pathlib import Path

import platformdirs
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VCBSettings(BaseSettings):
    """
    Centralizes all settings for the VCB project.

    Easily allows overwriting settings via environment variables and
    provides a shared "state" without passing parameters around.
    """

    # Configuration for the Pydantic model
    model_config = SettingsConfigDict(env_prefix="VCB_", extra="ignore", env_ignore_empty=True)

    # Constants
    txam_model_path: Path = Path("/rxrx/data/valence/hooke/predict/txam_checkpoints/TxAM_alpha/checkpoints")

    # Paths
    save_dir: Path | None = None
    cache_dir: Path = Field(default_factory=lambda: platformdirs.user_cache_dir(appname="vcb"))

    def ensure_save_dir(self, *subdirs: str) -> Path:
        """
        Return the save directory, potentially joined with one or more subdirectories.
        """
        if self.save_dir is None:
            raise ValueError("Save directory is not set.")

        save_dir = self.save_dir
        for subdir in subdirs:
            save_dir = save_dir / subdir

        return save_dir


settings = VCBSettings()
