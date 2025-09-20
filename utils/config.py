# utils/config.py - Configuration management for performance optimization

"""
Configuration management for the LLM Text Summarization Tool.

This module provides centralized configuration for performance optimization,
caching, and system settings.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # Performance settings
    "performance": {
        "cache_size_limit": 100,
        "cache_ttl": 3600,  # 1 hour
        "memory_threshold": 0.8,  # 80%
        "cleanup_interval": 300,  # 5 minutes
        "enable_background_cleanup": True,
        "enable_memory_monitoring": True,
        "enable_performance_tracking": True,
    },
    # Model settings
    "models": {
        "default_max_tokens": 1024,
        "chinese_max_tokens": 800,
        "enable_model_caching": True,
        "model_cache_ttl": 7200,  # 2 hours
        "lazy_loading": True,
    },
    # Processing settings
    "processing": {
        "max_file_size_mb": 10,
        "min_text_length": 50,
        "max_text_length": 100000,
        "chunk_overlap": 50,
        "batch_size": 5,
        "processing_delay": 0.1,
    },
    # UI settings
    "ui": {
        "enable_performance_dashboard": True,
        "enable_performance_alerts": True,
        "performance_data_retention_hours": 24,
        "max_performance_data_points": 1000,
        "performance_collection_interval": 5,
    },
    # Logging settings
    "logging": {
        "level": "INFO",
        "enable_performance_logging": True,
        "log_file": "performance.log",
        "max_log_size_mb": 10,
        "backup_count": 5,
    },
}

# Configuration file path
CONFIG_FILE = Path("config.json")


class ConfigManager:
    """Configuration manager for the application."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or CONFIG_FILE
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    loaded_config = json.load(f)
                    self._merge_config(self.config, loaded_config)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("No configuration file found, using defaults")
            self.save_config()

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> None:
        """Merge loaded configuration with defaults."""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.get("performance", {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get("models", {})

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.get("processing", {})

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self.get("ui", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})

    def update_performance_config(self, updates: Dict[str, Any]) -> None:
        """Update performance configuration."""
        perf_config = self.get_performance_config()
        perf_config.update(updates)
        self.set("performance", perf_config)

    def update_model_config(self, updates: Dict[str, Any]) -> None:
        """Update model configuration."""
        model_config = self.get_model_config()
        model_config.update(updates)
        self.set("models", model_config)

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = DEFAULT_CONFIG.copy()
        self.save_config()
        logger.info("Configuration reset to defaults")

    def export_config(self, file_path: Path) -> None:
        """Export configuration to a file."""
        try:
            with open(file_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")

    def import_config(self, file_path: Path) -> None:
        """Import configuration from a file."""
        try:
            with open(file_path, "r") as f:
                imported_config = json.load(f)
            self._merge_config(self.config, imported_config)
            self.save_config()
            logger.info(f"Configuration imported from {file_path}")
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")


# Environment variable overrides
def apply_env_overrides(config_manager: ConfigManager) -> None:
    """Apply environment variable overrides to configuration."""
    env_mappings = {
        "LLM_CACHE_SIZE_LIMIT": "performance.cache_size_limit",
        "LLM_CACHE_TTL": "performance.cache_ttl",
        "LLM_MEMORY_THRESHOLD": "performance.memory_threshold",
        "LLM_CLEANUP_INTERVAL": "performance.cleanup_interval",
        "LLM_MAX_FILE_SIZE_MB": "processing.max_file_size_mb",
        "LLM_MIN_TEXT_LENGTH": "processing.min_text_length",
        "LLM_MAX_TEXT_LENGTH": "processing.max_text_length",
        "LLM_LOG_LEVEL": "logging.level",
        "LLM_ENABLE_PERFORMANCE_DASHBOARD": "ui.enable_performance_dashboard",
        "LLM_ENABLE_PERFORMANCE_ALERTS": "ui.enable_performance_alerts",
    }

    for env_var, config_key in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                # Try to convert to appropriate type
                if env_value.lower() in ("true", "false"):
                    value = env_value.lower() == "true"
                elif env_value.isdigit():
                    value = int(env_value)
                elif env_value.replace(".", "").isdigit():
                    value = float(env_value)
                else:
                    value = env_value

                config_manager.set(config_key, value)
                logger.info(
                    f"Environment override: {env_var} -> {config_key} = {value}"
                )
            except Exception as e:
                logger.warning(f"Could not apply environment override {env_var}: {e}")


# Global configuration instance
config_manager = ConfigManager()
apply_env_overrides(config_manager)


# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return config_manager.get(key, default)


def set_config(key: str, value: Any) -> None:
    """Set configuration value."""
    config_manager.set(key, value)


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    return config_manager.get_performance_config()


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return config_manager.get_model_config()


def get_processing_config() -> Dict[str, Any]:
    """Get processing configuration."""
    return config_manager.get_processing_config()


def get_ui_config() -> Dict[str, Any]:
    """Get UI configuration."""
    return config_manager.get_ui_config()


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration."""
    return config_manager.get_logging_config()


def save_config() -> None:
    """Save current configuration."""
    config_manager.save_config()


def reset_config() -> None:
    """Reset configuration to defaults."""
    config_manager.reset_to_defaults()
