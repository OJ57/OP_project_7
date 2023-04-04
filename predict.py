import requests

url = "http://127.0.0.1:8000/predict"

# Replace this with the ID number for which you want to make a prediction
client_id = 181733

# Send a POST request to the API with the client ID
response = requests.post(url, json={"client_id": client_id})

# Print the result
print(response.json())
