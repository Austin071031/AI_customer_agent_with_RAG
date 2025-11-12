import os
print('DEEPSEEK_API_KEY:', os.getenv('DEEPSEEK_API_KEY'))
print('Current working directory:', os.getcwd())
print('Environment file exists:', os.path.exists('config/.env'))
