import requests
import os
import json

# --- IMPORTANT: Set your API key in your terminal before running ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # Fallback for easier testing if the environment variable isn't set.
    # Replace "YOUR_API_KEY_HERE" with your actual key.
    api_key = "AIzaSyDQ9SmrtAQ8P8WrWtO1ROPA2xgPSJgezKY" 
    if api_key == "AIzaSyDQ9SmrtAQ8P8WrWtO1ROPA2xgPSJgezKY":
      raise ValueError("Please replace 'YOUR_API_KEY_HERE' with your actual API key.")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"

headers = { 'Content-Type': 'application/json' }
data = {
    "contents": [{
        "parts": [{"text": "Hello, please respond with a single word: 'Success'"}]
    }]
}

print("Sending a basic test request to the Gemini API...")

try:
    response = requests.post(url, headers=headers, json=data, timeout=20)
    if response.status_code == 200:
        response_json = response.json()
        text_response = response_json['candidates'][0]['content']['parts'][0]['text']
        print("✅ API Call Successful!")
        print(f"   Response: {text_response.strip()}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print("   Response Body:")
        print(response.text)
except requests.exceptions.RequestException as e:
    print(f"A network error occurred: {e}")
