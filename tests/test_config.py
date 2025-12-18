import sys
sys.path.append('.')
from src.services.config_manager import ConfigManager

try:
    config = ConfigManager()
    print('Settings loaded successfully')
    print(f'API Key present: {bool(config.settings.deepseek.api_key)}')
    print(f'API Key length: {len(config.settings.deepseek.api_key)}')
    print(f'API Key value: {config.settings.deepseek.api_key}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
