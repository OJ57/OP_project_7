import streamlit as st
import pandas as pd
import httpx
import joblib
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap

# Configuration de l'API
# API_URL = "http://localhost:8000"
API_URL = "https://fastapi-project7.herokuapp.com"

ordered_features_list = joblib.load('ordered_features_list.joblib')
explainer = joblib.load('explainer.pkl')

st.set_page_config(
    page_title="Prédiction de remboursement de prêt",
    page_icon="logo.png",
    layout="wide"
)


# Chargez les données
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


@st.cache_data
def get_response(client_id):
    client_id = int(client_id)  # Convertir client_id en int
    response = httpx.post(f"{API_URL}/predict", json={"client_id": client_id})
    if response.status_code == 200:
        return response.json()
    else:
        return None


st.title("Prédiction de défaillance")
st.header("Vos clients vont-ils rembourser leur prêt ?")
st.divider()

data = load_data("test_API.csv")

st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.divider()

if st.sidebar.checkbox('**Afficher le jeu de données**'):
    st.subheader("Données des clients")
    st.write(data[["SK_ID_CURR"] + ordered_features_list])
    st.divider()

# Sélection du client_id
st.subheader("Choisissez un **numéro client** pour obtenir la prédiction de défaillance")
client_id = st.selectbox("**Numéro client**", data["SK_ID_CURR"].unique(), label_visibility="visible")

# Obtention la réponse de l'API
response = get_response(client_id)

if response is None:
    st.write("Erreur lors de communication avec l'API.")

p = response["probability_of_failure"]
p = round(100 * p)

# Create the gauge chart
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=p,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={
        'text': "<b><span style='color:gray'>Probabilité de défaillance</span></b>",
        'font': {'size': 24, 'family': 'Arial'}
    },
    gauge={
        'axis': {
            'range': [0, 100],
            'tickwidth': 3,
            'tickmode': 'array',
            'tickvals': [0, 25, 50, 75, 100],
            'ticktext': ['0', '25', '50', '75', '100'],
            'tickfont': {'size': 16},
        },
        'bar': {'color': 'white', 'thickness': 0.4, 'line': {'color': 'gray', 'width': 2}},
        'bgcolor': 'white',
        'borderwidth': 2,
        'bordercolor': 'gray',
        'steps': [
            {'range': [0, 25], 'color': 'Green'},
            {'range': [25, 50], 'color': 'LimeGreen'},
            {'range': [50, 75], 'color': 'Orange'},
            {'range': [75, 100], 'color': 'Red'}
        ],
    }
))

gauge.update_layout(
    width=600,  # Set the width in pixels
    height=400  # Set the height in pixels
)

col1, col2, col3 = st.columns([0.5, 0.3, 0.5])
with col1:
    st.plotly_chart(gauge)
with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # Texte d'accompagnement de la jauge
    if 0 <= p < 25:
        score_text = 'La probabilité de défaillance du client est très faible.'
        st.success(score_text)
    elif 25 <= p < 50:
        score_text = 'La probabilité de défaillance du client est faible.'
        st.success(score_text)
    elif 50 <= p < 75:
        score_text = 'La probabilité de défaillance du client est élevée.'
        st.warning(score_text)
    else:
        score_text = 'La probabilité de défaillance du client est très élevée !'
        st.error(score_text)

st.divider()

if st.sidebar.checkbox('**Afficher les données du client**'):
    st.markdown(f"**Données du client n° {client_id}**")

    data_client = data[data["SK_ID_CURR"] == client_id]
    st.write(data_client[["SK_ID_CURR"] + ordered_features_list])
    st.divider()

if st.sidebar.checkbox('**Comparer les données du client**'):
    selected_client_data = data[data["SK_ID_CURR"] == client_id].iloc[0]

    # Sélection de la variable
    st.subheader("Choisissez une variable pour comparer le client au reste des clients")
    data_client = data[data["SK_ID_CURR"] == client_id][ordered_features_list]
    variable = st.selectbox("**Variable**", data_client.columns, label_visibility="visible")

    # Plot the distribution for a specific variable
    plt.figure(figsize=(4.5, 3.5))
    sns.set(style="darkgrid", font_scale=0.8)
    sns.histplot(data=data, x=variable, kde=True)
    client_value = selected_client_data[variable]
    plt.axvline(selected_client_data[variable], color='r', linestyle='--', label=f"Client {client_id}")

    # Add a small offset to the x-coordinate of the text for better readability
    offset = 0.02 * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0])

    plt.text(client_value + offset, plt.gca().get_ylim()[1] * 0.9, f"{client_value:.2f}", color="r", fontsize=9)
    plt.legend(fontsize=9)

    # Display the plot in Streamlit
    st.pyplot(plt.gcf(), use_container_width=False)

    # Clear the current figure
    plt.clf()
    st.divider()

if st.sidebar.checkbox('**Expliquer la prédiction**'):

    shap_v = response['shap_values']

    # Sélection de la variable
    st.subheader("Choisissez le type de figure")
    list_fig = ['Decision', 'Waterfall']

    type_fig = st.selectbox("**Type**", list_fig, label_visibility="visible")

    sns.set(style="darkgrid", font_scale=0.8)

    if type_fig == 'Decision':
        shap.decision_plot(explainer.expected_value, np.array(shap_v), data[data["SK_ID_CURR"] == client_id].drop(
            columns='SK_ID_CURR').iloc[0], link='logit')

        plt.gcf().set_size_inches(5, 6)

        # Display the plot in Streamlit
        st.pyplot(plt.gcf(), use_container_width=False)

        # Clear the current figure
        plt.clf()

    if type_fig == 'Waterfall':

        shap_values_instance = shap.Explanation(
            values=np.array(shap_v)[0],
            base_values=explainer.expected_value,
            data=data[data["SK_ID_CURR"] == client_id].drop(columns='SK_ID_CURR').iloc[0],
            feature_names=data.drop(columns='SK_ID_CURR').columns.tolist())

        # Plot the SHAP waterfall plot for a specific instance
        shap.waterfall_plot(shap_values_instance, max_display=20)

        plt.gcf().set_size_inches(5, 6)

        # Display the plot in Streamlit
        st.pyplot(plt.gcf(), use_container_width=False)

        # Clear the current figure
        plt.clf()

    st.divider()


if st.sidebar.checkbox('**Clients similaires**'):
    st.subheader("Données des 5 clients les plus similaires")

    nearest_clients = response['nearest_clients']
    st.dataframe(data.loc[data['SK_ID_CURR'].isin(nearest_clients)]
                 [["SK_ID_CURR"] + ordered_features_list].style.highlight_max(axis=0))

    average_probability = response['average_probability']
    positive_cases = response['positive_cases']

    st.markdown(f"Average probability: {average_probability:.2f}")
    st.markdown(f"Number of positive cases: {positive_cases}")

    st.divider()













































