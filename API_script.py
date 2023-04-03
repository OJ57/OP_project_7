from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the pre-processed data
data = pd.read_csv("test_API.csv")

# Load the model
model = joblib.load("model_prediction.joblib")

# Load the explainer
explainer = joblib.load("explainer.pkl")


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

    # Make a prediction using the loaded model - probability
    prediction = model.predict_proba(features)[:, 1][0]

    # Make a prediction using the loaded model
    will_fail = 'yes' if int(prediction >= THRESHOLD) == 1 else 'no'

    # Calculate SHAP values for the given client
    shap_values = explainer(features, check_additivity=False)

    # Return the prediction
    return {"client_id": client_id,
            "probability_of_failure": prediction,
            "will_fail": will_fail,
            "shap_values": shap_values.values.tolist()
            }
