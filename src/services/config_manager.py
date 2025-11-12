"""
Configuration Manager for AI Customer Agent

This module provides a centralized configuration management system that handles
application settings from multiple sources (YAML files, environment variables)
with proper validation and encryption for sensitive data.

Key Features:
- Load settings from YAML configuration files
- Support environment variables for sensitive data
- Encrypt and decrypt sensitive configuration values
- Validate configuration using Pydantic models
- Provide type-safe access to configuration values
"""

import os
import base64
import hashlib
from typing import Any, Dict, Optional, List
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class ConfigManagerError(Exception):
    """
    Custom exception for configuration management errors.
    
    This exception is raised when configuration operations fail
    due to validation errors, file access issues, or other
    configuration-related problems.
    """
    pass

# Load environment variables from .env file
load_dotenv(dotenv_path=Path.cwd() / "config" / ".env")


class EncryptionManager:
    """
    Handles encryption and decryption of sensitive configuration data.
    
    Uses Fernet symmetric encryption with a key derived from a passphrase
    stored in environment variables.
    """
    
    def __init__(self, passphrase: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            passphrase: Encryption passphrase. If None, uses environment variable.
        """
        self.passphrase = passphrase or os.getenv("CONFIG_ENCRYPTION_KEY", "default-encryption-key")
        self.fernet = self._create_fernet()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet instance from passphrase."""
        # Derive key from passphrase using PBKDF2
        salt = b'ai_customer_agent_salt'  # In production, use a random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.passphrase.encode()))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Plain text data to encrypt
            
        Returns:
            Encrypted string
        """
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted string to decrypt
            
        Returns:
            Decrypted plain text
        """
        return self.fernet.decrypt(encrypted_data.encode()).decode()


class AppConfig(BaseModel):
    """Application-level configuration."""
    name: str = Field(default="AI Customer Agent")
    version: str = Field(default="1.0.0")
    description: str = Field(default="Local AI customer service agent")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")


class DeepSeekConfig(BaseModel):
    """DeepSeek API configuration."""
    base_url: str = Field(default="https://api.deepseek.com")
    model: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=4000)
    timeout: int = Field(default=30, ge=1)
    
    @property
    def api_key(self) -> str:
        """Get API key from environment variable with fallback."""
        return os.getenv("DEEPSEEK_API_KEY", "")
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is within reasonable range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base configuration."""
    persist_directory: str = Field(default="./knowledge_base/chroma_db")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    max_documents: int = Field(default=10000, ge=1)


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=3, ge=1)
    search_type: str = Field(default="similarity")


class FileProcessingConfig(BaseModel):
    """File processing configuration."""
    supported_formats: List[str] = Field(default=["pdf", "txt", "docx", "doc", "xlsx", "xls"])
    max_file_size_mb: int = Field(default=10, ge=1)


class ChatConfig(BaseModel):
    """Chat configuration."""
    max_history_length: int = Field(default=20, ge=1)
    enable_streaming: bool = Field(default=True)
    system_prompt: str = Field(default="You are a helpful customer service assistant.")


class GPUConfig(BaseModel):
    """GPU configuration."""
    enable_cuda: bool = Field(default=True)
    device: str = Field(default="cuda")
    max_memory_gb: int = Field(default=8, ge=1)


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=8000, ge=1024, le=65535)
    reload: bool = Field(default=True)
    workers: int = Field(default=1, ge=1)


class UIConfig(BaseModel):
    """Streamlit UI configuration."""
    port: int = Field(default=8501, ge=1024, le=65535)
    theme: str = Field(default="light")
    page_title: str = Field(default="AI Customer Agent")


class FullConfig(BaseModel):
    """Complete application configuration model."""
    app: AppConfig = Field(default_factory=AppConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    file_processing: FileProcessingConfig = Field(default_factory=FileProcessingConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


class ConfigManager:
    """
    Main configuration manager class.
    
    Handles loading, validation, and management of application settings
    from multiple sources with encryption support for sensitive data.
    """
    
    def __init__(self, config_path: str = "./config", env_file: str = ".env"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration directory
            env_file: Name of environment file
        """
        self.config_path = Path(config_path)
        self.env_file = self.config_path / env_file
        self.encryption_manager = EncryptionManager()
        self.settings: Optional[FullConfig] = None
        
        # Ensure config directory exists
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        # Load settings on initialization
        self.load_settings()
    
    def load_settings(self) -> FullConfig:
        """
        Load settings from all available sources.
        
        Returns:
            Validated configuration object
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration validation fails
        """
        # Load from YAML file
        yaml_settings = self._load_yaml_settings()
        
        # Merge with environment variables
        env_settings = self._load_env_settings()
        merged_settings = self._merge_settings(yaml_settings, env_settings)
        
        # Validate and create configuration object
        try:
            self.settings = FullConfig(**merged_settings)
            return self.settings
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _load_yaml_settings(self) -> Dict[str, Any]:
        """
        Load settings from YAML configuration file.
        
        Returns:
            Dictionary of settings from YAML file
            
        Raises:
            FileNotFoundError: If settings.yaml doesn't exist
        """
        yaml_file = self.config_path / "settings.yaml"
        
        if not yaml_file.exists():
            # Create default settings file if it doesn't exist
            self._create_default_settings(yaml_file)
        
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file) or {}
    
    def _load_env_settings(self) -> Dict[str, Any]:
        """
        Load settings from environment variables.
        
        Returns:
            Dictionary of settings from environment variables
        """
        env_settings = {}
        
        # DeepSeek API key from environment
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            env_settings["deepseek"] = {"api_key": deepseek_api_key}
        
        # Encryption key from environment
        encryption_key = os.getenv("CONFIG_ENCRYPTION_KEY")
        if encryption_key:
            env_settings["encryption"] = {"key": encryption_key}
        
        return env_settings
    
    def _merge_settings(self, yaml_settings: Dict[str, Any], env_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge settings from YAML and environment variables.
        
        Args:
            yaml_settings: Settings from YAML file
            env_settings: Settings from environment variables
            
        Returns:
            Merged settings dictionary
        """
        merged = yaml_settings.copy()
        
        # Deep merge environment settings
        for key, value in env_settings.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        
        return merged
    
    def _create_default_settings(self, yaml_file: Path) -> None:
        """
        Create default settings file if it doesn't exist.
        
        Args:
            yaml_file: Path to YAML settings file
        """
        default_settings = {
            "app": {
                "name": "AI Customer Agent",
                "version": "1.0.0",
                "description": "Local AI customer service agent with DeepSeek API integration",
                "debug": False,
                "log_level": "INFO"
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 30
            },
            "knowledge_base": {
                "persist_directory": "./knowledge_base/chroma_db",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_documents": 10000
            },
            "vector_search": {
                "similarity_threshold": 0.7,
                "max_results": 3,
                "search_type": "similarity"
            },
            "file_processing": {
                "supported_formats": ["pdf", "txt", "docx", "doc", "xlsx", "xls"],
                "max_file_size_mb": 10
            },
            "chat": {
                "max_history_length": 20,
                "enable_streaming": True,
                "system_prompt": "You are a helpful customer service assistant. Provide accurate and friendly responses to customer queries."
            },
            "gpu": {
                "enable_cuda": True,
                "device": "cuda",
                "max_memory_gb": 8
            },
            "api": {
                "host": "localhost",
                "port": 8000,
                "reload": True,
                "workers": 1
            },
            "ui": {
                "port": 8501,
                "theme": "light",
                "page_title": "AI Customer Agent"
            }
        }
        
        with open(yaml_file, 'w', encoding='utf-8') as file:
            yaml.dump(default_settings, file, default_flow_style=False, indent=2)
    
    def get_setting(self, key: str) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Dot-separated key path (e.g., "deepseek.api_key")
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If the key doesn't exist
        """
        if not self.settings:
            raise RuntimeError("Settings not loaded. Call load_settings() first.")
        
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                raise KeyError(f"Configuration key '{key}' not found")
        
        return value
    
    def update_setting(self, key: str, value: Any) -> bool:
        """
        Update a configuration setting.
        
        Args:
            key: Dot-separated key path
            value: New value to set
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            if not self.settings:
                return False
            
            keys = key.split('.')
            current = self.settings
            
            # Navigate to the parent object
            for k in keys[:-1]:
                if hasattr(current, k):
                    current = getattr(current, k)
                else:
                    return False
            
            # Set the value on the parent object
            setattr(current, keys[-1], value)
            return True
            
        except (AttributeError, ValueError):
            return False
    
    def save_settings(self) -> bool:
        """
        Save current settings to YAML file.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            if not self.settings:
                return False
            
            yaml_file = self.config_path / "settings.yaml"
            settings_dict = self.settings.model_dump()
            
            with open(yaml_file, 'w', encoding='utf-8') as file:
                yaml.dump(settings_dict, file, default_flow_style=False, indent=2)
            
            return True
            
        except Exception:
            return False
    
    def encrypt_value(self, value: str) -> str:
        """
        Encrypt a sensitive value.
        
        Args:
            value: Plain text value to encrypt
            
        Returns:
            Encrypted string
        """
        return self.encryption_manager.encrypt(value)
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt an encrypted value.
        
        Args:
            encrypted_value: Encrypted string to decrypt
            
        Returns:
            Decrypted plain text
        """
        return self.encryption_manager.decrypt(encrypted_value)
    
    def get_deepseek_api_key(self) -> str:
        """
        Get DeepSeek API key with decryption if needed.

        Returns:
            Decrypted API key string
        """
        # API key is stored in environment variables, not encrypted in this implementation
        # but we provide the method for consistency
        return self.settings.deepseek.api_key if self.settings else ""

    def get_api_config(self):
        """
        Get API configuration for the endpoint.

        Returns:
            APIConfig instance compatible with config_models
        """
        if not self.settings:
            raise ConfigManagerError("Settings not loaded")
        
        # Import the model from config_models to avoid circular imports
        from ..models.config_models import APIConfig
        
        # Map from DeepSeekConfig to APIConfig
        return APIConfig(
            api_key=self.settings.deepseek.api_key,
            base_url=self.settings.deepseek.base_url,
            model=self.settings.deepseek.model,
            temperature=self.settings.deepseek.temperature,
            max_tokens=self.settings.deepseek.max_tokens
        )

    def get_db_config(self):
        """
        Get database configuration for the endpoint.

        Returns:
            DatabaseConfig instance compatible with config_models
        """
        if not self.settings:
            raise ConfigManagerError("Settings not loaded")
        
        # Import the model from config_models to avoid circular imports
        from ..models.config_models import DatabaseConfig
        
        # Map from KnowledgeBaseConfig to DatabaseConfig
        return DatabaseConfig(
            persist_directory=self.settings.knowledge_base.persist_directory,
            collection_name="documents"  # Default collection name
        )

    def get_app_config(self):
        """
        Get application configuration for the endpoint.

        Returns:
            AppConfig instance compatible with config_models
        """
        if not self.settings:
            raise ConfigManagerError("Settings not loaded")
        
        # Import the model from config_models to avoid circular imports
        from ..models.config_models import AppConfig
        
        # Convert log_level to lowercase to match the expected pattern
        log_level = self.settings.app.log_level.lower()
        
        # Create AppConfig with the required structure
        return AppConfig(
            api_config=self.get_api_config(),
            db_config=self.get_db_config(),
            log_level=log_level,
            enable_gpu=self.settings.gpu.enable_cuda,
            max_conversation_history=self.settings.chat.max_history_length
        )

    def update_api_config(self, update_data: dict) -> bool:
        """
        Update API configuration with provided data.

        Args:
            update_data: Dictionary with API configuration updates

        Returns:
            True if update was successful, False otherwise
        """
        try:
            if not self.settings:
                return False

            # Update DeepSeek configuration
            deepseek_updates = {}
            if "api_key" in update_data:
                # Note: API key is stored in environment, so we can't update it here
                # In a real implementation, you might update the environment variable
                pass
            if "base_url" in update_data:
                deepseek_updates["base_url"] = update_data["base_url"]
            if "model" in update_data:
                deepseek_updates["model"] = update_data["model"]
            if "temperature" in update_data:
                deepseek_updates["temperature"] = update_data["temperature"]
            if "max_tokens" in update_data:
                deepseek_updates["max_tokens"] = update_data["max_tokens"]

            # Apply updates to the settings
            for key, value in deepseek_updates.items():
                if hasattr(self.settings.deepseek, key):
                    setattr(self.settings.deepseek, key, value)

            # Save the updated settings
            return self.save_settings()

        except Exception:
            return False

    def update_db_config(self, update_data: dict) -> bool:
        """
        Update database configuration with provided data.

        Args:
            update_data: Dictionary with database configuration updates

        Returns:
            True if update was successful, False otherwise
        """
        try:
            if not self.settings:
                return False

            # Update knowledge base configuration
            kb_updates = {}
            if "persist_directory" in update_data:
                kb_updates["persist_directory"] = update_data["persist_directory"]
            if "collection_name" in update_data:
                # Note: collection_name is not in KnowledgeBaseConfig, but we'll handle it
                pass

            # Apply updates to the settings
            for key, value in kb_updates.items():
                if hasattr(self.settings.knowledge_base, key):
                    setattr(self.settings.knowledge_base, key, value)

            # Save the updated settings
            return self.save_settings()

        except Exception:
            return False

    def update_app_config(self, update_data: dict) -> bool:
        """
        Update application configuration with provided data.

        Args:
            update_data: Dictionary with application configuration updates

        Returns:
            True if update was successful, False otherwise
        """
        try:
            if not self.settings:
                return False

            # Update application configuration
            app_updates = {}
            if "log_level" in update_data:
                app_updates["log_level"] = update_data["log_level"]
            if "enable_gpu" in update_data:
                self.settings.gpu.enable_cuda = update_data["enable_gpu"]
            if "max_conversation_history" in update_data:
                self.settings.chat.max_history_length = update_data["max_conversation_history"]

            # Apply updates to the settings
            for key, value in app_updates.items():
                if hasattr(self.settings.app, key):
                    setattr(self.settings.app, key, value)

            # Save the updated settings
            return self.save_settings()

        except Exception:
            return False

    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to default values.

        Returns:
            True if reset was successful, False otherwise
        """
        try:
            # Create a new ConfigManager instance to get default settings
            default_manager = ConfigManager()
            self.settings = default_manager.settings
            return self.save_settings()
        except Exception:
            return False

    def validate_configuration(self) -> dict:
        """
        Validate the current configuration.

        Returns:
            Dictionary with validation results (valid, errors, warnings)
        """
        errors = []
        warnings = []

        try:
            # Validate API configuration
            if not self.settings.deepseek.api_key:
                warnings.append("DeepSeek API key is not configured")

            # Validate database configuration
            if not self.settings.knowledge_base.persist_directory:
                errors.append("Knowledge base persist directory is not configured")

            # Validate application configuration
            if not self.settings.app.log_level:
                warnings.append("Log level is not configured, using default")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Configuration validation failed: {str(e)}"],
                "warnings": []
            }

    def create_backup(self) -> dict:
        """
        Create a backup of the current configuration.

        Returns:
            Dictionary with backup information
        """
        try:
            from datetime import datetime
            import shutil
            
            # Create backup directory if it doesn't exist
            backup_dir = self.config_path / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"settings_backup_{timestamp}.yaml"
            
            # Copy current settings file to backup
            settings_file = self.config_path / "settings.yaml"
            if settings_file.exists():
                shutil.copy2(settings_file, backup_file)
            
            config_files = []
            if self.config_path.exists():
                for file in self.config_path.iterdir():
                    if file.is_file() and file.suffix in ['.yaml', '.yml', '.env']:
                        config_files.append(str(file))
            
            return {
                "backup_file": str(backup_file),
                "backup_time": datetime.now().isoformat(),
                "config_files": config_files
            }
            
        except Exception as e:
            return {
                "backup_file": None,
                "backup_time": None,
                "config_files": [],
                "error": str(e)
            }

    def get_last_modified(self) -> str:
        """
        Get last modification timestamp.

        Returns:
            ISO format timestamp string
        """
        # For now, return current time. In production, this would track actual file modification.
        from datetime import datetime
        return datetime.now().isoformat()

    def health_check(self) -> bool:
        """
        Health check for configuration manager.

        Returns:
            True if healthy, False otherwise
        """
        try:
            return self.settings is not None and isinstance(self.settings, FullConfig)
        except Exception:
            return False

    def get_loaded_config_files(self) -> List[str]:
        """
        Get list of loaded configuration files.

        Returns:
            List of file paths
        """
        files = []
        if self.config_path.exists():
            for file in self.config_path.iterdir():
                if file.is_file() and file.suffix in ['.yaml', '.yml', '.env']:
                    files.append(str(file))
        return files


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get or create the global configuration manager instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_settings() -> FullConfig:
    """
    Get the current application settings.
    
    Returns:
        FullConfig object with current settings
    """
    return get_config_manager().settings
