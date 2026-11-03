"""
GPULivre User Configuration & Toggle Mechanism.
Allows the user of AssistentePessoal to enable or disable GPULivre orchestration
and Hybrid Fallback (Colab / Gemini) at runtime with persistent state.
"""

from __future__ import annotations
import os
import json
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("gpulivre.toggle")

CONFIG_PATH = os.path.expanduser("~/.antigravity/gpulivre_user_config.json")


class GPULivreUserConfig(BaseModel):
    gpulivre_enabled: bool = Field(
        default=True,
        description="Chave mestra: ativa ou desativa o gerenciamento de VRAM e preempção do GPULivre."
    )
    hybrid_fallback_enabled: bool = Field(
        default=True,
        description="Ativa ou desativa o desvio automático para Colab / Gemini quando a GPU local estiver saturada."
    )
    preferred_fallback: str = Field(
        default="auto",
        description="'auto' | 'colab' | 'gemini' | 'cpu'"
    )
    safety_headroom_mb: float = Field(
        default=512.0,
        description="Margem de segurança de VRAM livre intocável para estabilidade do SO."
    )
    notify_on_preemption: bool = Field(
        default=True,
        description="Emite aviso no console/voz quando um processo P3 for preemptado por P1."
    )


class UserConfigManager:
    """Manages reading, persisting, and toggling user preferences for GPULivre."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self._config = self._load()

    def _load(self) -> GPULivreUserConfig:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return GPULivreUserConfig.model_validate(data)
            except Exception as e:
                logger.error(f"Failed to read {self.config_path}: {e}. Using defaults.")
        # Create default config file if not exists
        default_cfg = GPULivreUserConfig()
        self.save(default_cfg)
        return default_cfg

    def save(self, config: Optional[GPULivreUserConfig] = None) -> None:
        if config:
            self._config = config
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                f.write(self._config.model_dump_json(indent=2))
            logger.info(f"GPULivre user config updated at {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to write user config to {self.config_path}: {e}")

    @property
    def config(self) -> GPULivreUserConfig:
        return self._config

    def is_gpulivre_enabled(self) -> bool:
        # Check environment variable override first (GPULIVRE_ENABLED=0/1)
        env_val = os.getenv("GPULIVRE_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes")
        return self._config.gpulivre_enabled

    def is_fallback_enabled(self) -> bool:
        env_val = os.getenv("GPULIVRE_FALLBACK_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes")
        return self._config.hybrid_fallback_enabled

    def set_gpulivre_enabled(self, enabled: bool) -> GPULivreUserConfig:
        self._config.gpulivre_enabled = enabled
        self.save()
        logger.info(f"User set gpulivre_enabled = {enabled}")
        return self._config

    def set_fallback_enabled(self, enabled: bool) -> GPULivreUserConfig:
        self._config.hybrid_fallback_enabled = enabled
        self.save()
        logger.info(f"User set hybrid_fallback_enabled = {enabled}")
        return self._config


# Global singleton instance
user_config_manager = UserConfigManager()
