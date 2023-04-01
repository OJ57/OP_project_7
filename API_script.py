from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the pre-processed data
data = pd.read_csv("test_API.csv")  # Replace with the path to your CSV file
# data = data.set_index("client_id")  # Assuming the column name is "client_id"

# Load the model
model = joblib.load("model_prediction.joblib")

app = FastAPI()


class ClientID(BaseModel):
    client_id: int


@app.post("/predict")
async def predict_failure(client: ClientID):
    client_id = client.client_id

    if client_id not in data.index:
        print(client_id)
        raise HTTPException(status_code=404, detail=f"Client ID {client_id} not found")

    # Get the features for the given client_id
    features = data.loc[client_id].values.reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = model.predict_proba(features)[:, 1]

    # Return the prediction
    return {"client_id": client_id, "probability_of_failure": prediction[0]}
