from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the pre-processed data
data = pd.read_csv("test_API.csv")  # Replace with the path to your CSV file

# Load the model
model = joblib.load("model_prediction.joblib")

app = FastAPI()

THRESHOLD = 0.52


class ClientID(BaseModel):
    client_id: int


@app.post("/predict")
async def predict_failure(client: ClientID):
    client_id = client.client_id

    if client_id not in data['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail=f"Client ID {client_id} not found")

    # Get the features for the given client_id
    features = data.loc[data['SK_ID_CURR'] == client_id].drop(columns='SK_ID_CURR').values.reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = model.predict_proba(features)[:, 1][0]

    will_fail = 'yes' if int(prediction >= THRESHOLD) == 1 else 'no'

    # Return the prediction
    return {"client_id": client_id, "probability_of_failure": prediction, "will_fail": will_fail}
