from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent / 'config' / '.env'
print('Environment file path:', env_path)
print('Environment file exists:', env_path.exists())

load_dotenv(dotenv_path=env_path)
print('DEEPSEEK_API_KEY after load:', os.getenv('DEEPSEEK_API_KEY'))
