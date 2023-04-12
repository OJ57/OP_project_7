from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the pre-processed data
data = pd.read_csv("test_API.csv")

# Load the model
model = joblib.load("model_prediction.joblib")

# Load the explainer
explainer = joblib.load("explainer.pkl")

app = FastAPI()

THRESHOLD = 0.5


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

    # Compute the 5 nearest clients
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(data.drop(columns='SK_ID_CURR').values)
    _, indices = nbrs.kneighbors(features)

    nearest_clients = data.iloc[indices[0]]['SK_ID_CURR'].values.tolist()

    # Get probabilities of the nearest clients
    probabilities = []

    for nearest_client_id in nearest_clients:

        features = data.loc[data['SK_ID_CURR'] == nearest_client_id].drop(columns='SK_ID_CURR').values.reshape(1, -1)

        prediction_nearest_client_id = model.predict_proba(features)[:, 1][0]

        probabilities.append(prediction_nearest_client_id)

    # Compute the average probability
    average_probability = sum(probabilities) / len(probabilities)

    # Count positive cases (above threshold)
    positive_cases = sum(1 for prob in probabilities if prob >= 0.5)

    # Get the features for the nearest clients
    features_nearest_clients = data.loc[data['SK_ID_CURR'].isin(nearest_clients)].drop(columns='SK_ID_CURR').values

    # Calculate SHAP values for the nearest clients
    shap_values_API_nearest_clients = explainer(features_nearest_clients, check_additivity=False)

    return {"client_id": client_id,
            "probability_of_failure": prediction,
            "will_fail": will_fail,
            "shap_values": shap_values.values.tolist(),
            "nearest_clients": nearest_clients,
            "average_probability": average_probability,
            "positive_cases": positive_cases,
            "shap_values_nearest": shap_values_API_nearest_clients.values.tolist()
            }
