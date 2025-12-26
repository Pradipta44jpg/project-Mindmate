
import requests

# URL must match the Flask route exactly
url = "http://127.0.0.1:5000/predict"

# Example text to send for emotion prediction
data = {"text": "I am feeling very sad today."}

try:
    # Send POST request
    response = requests.post(url, json=data)

    # Print status code
    print("Status Code:", response.status_code)

    # Handle different response cases
    if response.status_code == 404:
        print("❌ 404 Not Found: Check your Flask route and URL.")
    elif response.status_code >= 400:
        print(f"❌ Server error: {response.status_code}")
        print("Response Text:", response.text)
    else:
        # Try to parse JSON
        try:
            result = response.json()
            print("✅ Response JSON:", result)
        except requests.exceptions.JSONDecodeError:
            print("❌ Response is not JSON!")
            print("Raw Response Text:", response.text)

except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to Flask server. Make sure it is running.")
except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
