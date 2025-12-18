import requests
import sys

try:
    response = requests.get('http://localhost:8001/health', timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
