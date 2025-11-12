"""
Unit tests for the Configuration Manager module.

This test suite covers all functionality of the ConfigManager class,
including settings loading, validation, encryption, and error handling.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.services.config_manager import (
    ConfigManager,
    EncryptionManager,
    AppConfig,
    DeepSeekConfig,
    KnowledgeBaseConfig,
    VectorSearchConfig,
    FileProcessingConfig,
    ChatConfig,
    GPUConfig,
    APIConfig,
    UIConfig,
    FullConfig,
    get_config_manager,
    get_settings
)


class TestEncryptionManager:
    """Test cases for EncryptionManager class."""
    
    def test_encryption_manager_initialization(self):
        """Test that EncryptionManager initializes correctly."""
        # Test with default passphrase
        manager = EncryptionManager()
        assert manager.passphrase == "default-encryption-key"
        assert manager.fernet is not None
        
        # Test with custom passphrase
        custom_manager = EncryptionManager("custom-passphrase")
        assert custom_manager.passphrase == "custom-passphrase"
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption work correctly together."""
        manager = EncryptionManager("test-passphrase")
        original_data = "sensitive-api-key-12345"
        
        # Encrypt the data
        encrypted = manager.encrypt(original_data)
        
        # Verify encrypted data is different from original
        assert encrypted != original_data
        assert isinstance(encrypted, str)
        
        # Decrypt the data
        decrypted = manager.decrypt(encrypted)
        
        # Verify decrypted data matches original
        assert decrypted == original_data
    
    def test_different_passphrases_produce_different_encryption(self):
        """Test that different passphrases produce different encrypted results."""
        manager1 = EncryptionManager("passphrase1")
        manager2 = EncryptionManager("passphrase2")
        original_data = "same-data"
        
        encrypted1 = manager1.encrypt(original_data)
        encrypted2 = manager2.encrypt(original_data)
        
        # Verify different passphrases produce different encrypted results
        assert encrypted1 != encrypted2


class TestConfigModels:
    """Test cases for Pydantic configuration models."""
    
    def test_app_config_defaults(self):
        """Test AppConfig with default values."""
        config = AppConfig()
        assert config.name == "AI Customer Agent"
        assert config.version == "1.0.0"
        assert config.debug is False
        assert config.log_level == "INFO"
    
    def test_deepseek_config_validation(self):
        """Test DeepSeekConfig validation rules."""
        # Test valid temperature
        config = DeepSeekConfig(temperature=1.5)
        assert config.temperature == 1.5
        
        # Test temperature validation
        with pytest.raises(ValueError):
            DeepSeekConfig(temperature=2.5)
        
        # Test max_tokens validation
        with pytest.raises(ValueError):
            DeepSeekConfig(max_tokens=0)
    
    def test_knowledge_base_config_validation(self):
        """Test KnowledgeBaseConfig validation rules."""
        # Test valid chunk_size
        config = KnowledgeBaseConfig(chunk_size=500)
        assert config.chunk_size == 500
        
        # Test invalid chunk_size
        with pytest.raises(ValueError):
            KnowledgeBaseConfig(chunk_size=50)
    
    def test_api_config_port_validation(self):
        """Test APIConfig port validation."""
        # Test valid port
        config = APIConfig(port=8080)
        assert config.port == 8080
        
        # Test invalid port (too low)
        with pytest.raises(ValueError):
            APIConfig(port=80)
        
        # Test invalid port (too high)
        with pytest.raises(ValueError):
            APIConfig(port=70000)
    
    def test_full_config_creation(self):
        """Test FullConfig creation with all sub-configs."""
        config = FullConfig()
        assert isinstance(config.app, AppConfig)
        assert isinstance(config.deepseek, DeepSeekConfig)
        assert isinstance(config.knowledge_base, KnowledgeBaseConfig)
        assert isinstance(config.vector_search, VectorSearchConfig)
        assert isinstance(config.file_processing, FileProcessingConfig)
        assert isinstance(config.chat, ChatConfig)
        assert isinstance(config.gpu, GPUConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.ui, UIConfig)


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for test configurations
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir)
        
        # Create test settings file
        self.settings_file = self.config_path / "settings.yaml"
        self.test_settings = {
            "app": {
                "name": "Test AI Agent",
                "version": "1.0.0",
                "debug": True,
                "log_level": "DEBUG"
            },
            "deepseek": {
                "base_url": "https://api.test.deepseek.com",
                "model": "test-model",
                "temperature": 0.8,
                "max_tokens": 1000
            }
        }
        
        # Write test settings to file
        import yaml
        with open(self.settings_file, 'w') as f:
            yaml.dump(self.test_settings, f)
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager(config_path=str(self.config_path))
        assert manager.config_path == self.config_path
        assert manager.encryption_manager is not None
        assert manager.settings is not None
    
    def test_load_settings_from_yaml(self):
        """Test loading settings from YAML file."""
        manager = ConfigManager(config_path=str(self.config_path))
        settings = manager.settings
        
        # Verify settings were loaded from YAML
        assert settings.app.name == "Test AI Agent"
        assert settings.app.debug is True
        assert settings.app.log_level == "DEBUG"
        assert settings.deepseek.base_url == "https://api.test.deepseek.com"
        assert settings.deepseek.temperature == 0.8
    
    def test_create_default_settings(self):
        """Test creation of default settings file when it doesn't exist."""
        # Remove existing settings file
        self.settings_file.unlink()
        
        # Initialize config manager - should create default settings
        manager = ConfigManager(config_path=str(self.config_path))
        
        # Verify default settings file was created
        assert self.settings_file.exists()
        
        # Verify default settings were loaded
        assert manager.settings is not None
        assert manager.settings.app.name == "AI Customer Agent"
    
    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-api-key-12345"})
    def test_load_env_settings(self):
        """Test loading settings from environment variables."""
        manager = ConfigManager(config_path=str(self.config_path))
        
        # Verify environment variable was loaded
        api_key = manager.settings.deepseek.api_key
        assert api_key == "test-api-key-12345"
    
    def test_get_setting(self):
        """Test getting individual settings by key path."""
        manager = ConfigManager(config_path=str(self.config_path))
        
        # Test getting nested settings
        app_name = manager.get_setting("app.name")
        assert app_name == "Test AI Agent"
        
        temperature = manager.get_setting("deepseek.temperature")
        assert temperature == 0.8
        
        # Test getting non-existent setting
        with pytest.raises(KeyError):
            manager.get_setting("non.existent.key")
    
    def test_update_setting(self):
        """Test updating configuration settings."""
        manager = ConfigManager(config_path=str(self.config_path))
        
        # Update a setting
        success = manager.update_setting("app.name", "Updated Name")
        assert success is True
        assert manager.settings.app.name == "Updated Name"
        
        # Update nested setting
        success = manager.update_setting("deepseek.temperature", 0.9)
        assert success is True
        assert manager.settings.deepseek.temperature == 0.9
        
        # Test updating non-existent setting
        success = manager.update_setting("non.existent.key", "value")
        assert success is False
    
    def test_save_settings(self):
        """Test saving settings to YAML file."""
        manager = ConfigManager(config_path=str(self.config_path))
        
        # Update a setting
        manager.update_setting("app.name", "Saved Name")
        
        # Save settings
        success = manager.save_settings()
        assert success is True
        
        # Verify settings were saved
        import yaml
        with open(self.settings_file, 'r') as f:
            saved_settings = yaml.safe_load(f)
        
        assert saved_settings["app"]["name"] == "Saved Name"
    
    def test_encrypt_decrypt_values(self):
        """Test encryption and decryption of sensitive values."""
        manager = ConfigManager(config_path=str(self.config_path))
        
        original_value = "very-secret-api-key"
        
        # Encrypt the value
        encrypted = manager.encrypt_value(original_value)
        assert encrypted != original_value
        
        # Decrypt the value
        decrypted = manager.encrypt_value(original_value)
        assert decrypted != original_value  # Note: This is actually encrypting again
        
        # Test actual decryption
        actually_decrypted = manager.decrypt_value(encrypted)
        assert actually_decrypted == original_value
    
    def test_get_deepseek_api_key(self):
        """Test getting DeepSeek API key."""
        manager = ConfigManager(config_path=str(self.config_path))
        
        # Test without environment variable
        api_key = manager.get_deepseek_api_key()
        assert api_key == ""  # Empty string when no env var set
        
        # Test with environment variable
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            manager.load_settings()  # Reload to pick up env var
            api_key = manager.get_deepseek_api_key()
            assert api_key == "test-key"


class TestGlobalFunctions:
    """Test cases for global configuration functions."""
    
    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns a singleton instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
    
    def test_get_settings(self):
        """Test that get_settings returns the current settings."""
        settings = get_settings()
        assert isinstance(settings, FullConfig)
    
    def test_config_manager_with_custom_path(self):
        """Test ConfigManager with custom configuration path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_manager = ConfigManager(config_path=temp_dir)
            assert custom_manager.config_path == Path(temp_dir)
            
            # Verify it creates default settings when none exist
            settings_file = Path(temp_dir) / "settings.yaml"
            assert settings_file.exists()


class TestErrorHandling:
    """Test cases for error handling in configuration manager."""
    
    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "settings.yaml"
            
            # Write invalid YAML
            with open(settings_file, 'w') as f:
                f.write("invalid: yaml: content: [unclosed list")
            
            # Should raise an exception when loading
            with pytest.raises(Exception):
                ConfigManager(config_path=temp_dir)
    
    def test_missing_config_directory(self):
        """Test that missing config directory is created automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = Path(temp_dir) / "non_existent_config"
            
            # Should create the directory
            manager = ConfigManager(config_path=str(non_existent_dir))
            assert non_existent_dir.exists()
    
    def test_get_setting_without_loading(self):
        """Test error when getting setting before loading."""
        manager = ConfigManager.__new__(ConfigManager)  # Create without calling __init__
        manager.settings = None
        
        with pytest.raises(RuntimeError, match="Settings not loaded"):
            manager.get_setting("app.name")


class TestIntegration:
    """Integration tests for configuration management."""
    
    def test_complete_configuration_flow(self):
        """Test complete configuration loading and management flow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create custom settings
            custom_settings = {
                "app": {
                    "name": "Integration Test Agent",
                    "version": "2.0.0",
                    "debug": True
                },
                "deepseek": {
                    "model": "deepseek-coder",
                    "temperature": 0.3
                },
                "knowledge_base": {
                    "chunk_size": 1500,
                    "chunk_overlap": 300
                }
            }
            
            # Write settings to file
            import yaml
            settings_file = Path(temp_dir) / "settings.yaml"
            with open(settings_file, 'w') as f:
                yaml.dump(custom_settings, f)
            
            # Initialize config manager
            manager = ConfigManager(config_path=temp_dir)
            
            # Verify settings were loaded
            assert manager.settings.app.name == "Integration Test Agent"
            assert manager.settings.deepseek.model == "deepseek-coder"
            assert manager.settings.knowledge_base.chunk_size == 1500
            
            # Update a setting
            manager.update_setting("app.name", "Updated Integration Agent")
            
            # Save settings
            manager.save_settings()
            
            # Reload settings to verify persistence
            manager2 = ConfigManager(config_path=temp_dir)
            assert manager2.settings.app.name == "Updated Integration Agent"
    
    def test_environment_variable_override(self):
        """Test that environment variables override YAML settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create settings file
            import yaml
            settings_file = Path(temp_dir) / "settings.yaml"
            with open(settings_file, 'w') as f:
                yaml.dump({"app": {"name": "YAML Name"}}, f)
            
            # Set environment variable
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-api-key"}):
                manager = ConfigManager(config_path=temp_dir)
                
                # YAML setting should be preserved
                assert manager.settings.app.name == "YAML Name"
                
                # Environment variable should be available
                assert manager.settings.deepseek.api_key == "env-api-key"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
