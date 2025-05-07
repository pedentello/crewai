import requests
import json

url = 'http://127.0.0.1/blog'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}
data = {
    'topic': 'Agentic AI - Enterprise architecture.'
}

response = requests.post(url, headers=headers, json=data)

print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(json.dumps(response.json(), indent=4, ensure_ascii=False))
