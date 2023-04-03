import httpx
import pytest
import joblib
import pandas as pd
import numpy as np

# URL de l'API, http://localhost:8000 si local
API_URL = "http://localhost:8000"

THRESHOLD = 0.52


def test_api_valid_client_id():
    client_id = 181733  # client_id valide présent dans le fichier CSV

    response = httpx.post(f"{API_URL}/predict", json={"client_id": client_id})

    assert response.status_code == 200
    assert "client_id" in response.json()
    assert "probability_of_failure" in response.json()
    assert "will_fail" in response.json()


def test_api_invalid_client_id():
    client_id = -1  # client_id invalide qui n'est pas présent dans le fichier CSV

    response = httpx.post(f"{API_URL}/predict", json={"client_id": client_id})

    assert response.status_code == 404


def test_api_probability_calculation():
    client_id = 181733

    # Load the pre-processed data
    data = pd.read_csv("test_API.csv")

    # Load the model
    model = joblib.load("model_prediction.joblib")

    features = data.loc[data['SK_ID_CURR'] == client_id].drop(columns='SK_ID_CURR').values.reshape(1, -1)
    manually_calculated_probability = model.predict_proba(features)[:, 1][0]

    response = httpx.post(f"{API_URL}/predict", json={"client_id": client_id})

    assert abs(response.json()["probability_of_failure"] - manually_calculated_probability) < 1e-6


def test_api_decision_threshold():
    client_id = 181733

    response = httpx.post(f"{API_URL}/predict", json={"client_id": client_id})

    probability_of_failure = response.json()["probability_of_failure"]
    will_fail = response.json()["will_fail"]

    assert will_fail == 'yes' if probability_of_failure >= THRESHOLD else 'no'


def test_api_shap():
    client_id = 181733

    # Load the pre-processed data
    data = pd.read_csv("test_API.csv")

    # Load the explainer
    explainer = joblib.load("explainer.pkl")

    features = data.loc[data['SK_ID_CURR'] == client_id].drop(columns='SK_ID_CURR').values.reshape(1, -1)

    # Calculate SHAP values for the given client
    manually_calculated_shap_values = explainer(features, check_additivity=False)

    response = httpx.post(f"{API_URL}/predict", json={"client_id": client_id})
    response_shap_values = np.array(response.json()["shap_values"])

    assert np.allclose(response_shap_values, manually_calculated_shap_values.values, rtol=1e-6, atol=1e-6)
