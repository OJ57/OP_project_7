import httpx

API_URL = "https://fastapi-project7.herokuapp.com"

client_id = 450148

# Send a POST request with the client_id as input
response = httpx.post(f"{API_URL}/predict", json={"client_id": client_id})

# Check if the request was successful (status code 200)
if response.status_code == 200:

    prediction_result = response.json()
    print("Prediction result:", prediction_result)

else:

    print("Error:", response.status_code, response.text)
