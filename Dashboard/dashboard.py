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
    temp = data[["SK_ID_CURR"] + ordered_features_list]
    temp["SK_ID_CURR"] = temp["SK_ID_CURR"].apply(lambda x: '{:,.0f}'.format(x).replace(',', ''))
    st.subheader("Données des clients")
    st.dataframe(temp)
    st.divider()

# Sélection du client_id
st.subheader("Choisissez un **numéro client** pour obtenir la prédiction de défaillance")
client_id = st.selectbox("**Numéro client**", data["SK_ID_CURR"].unique(), index=4)

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

col1, col2, col3 = st.columns([0.5, 0.35, 0.5])
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
        score_text = 'La probabilité de défaillance du client est _très faible_. **Prêt accordé**'
        st.success(score_text)
    elif 25 <= p < 50:
        score_text = 'La probabilité de défaillance du client est _faible_. **Prêt accordé**'
        st.success(score_text)
    elif 50 <= p < 75:
        score_text = 'La probabilité de défaillance du client est _élevée_. **Prêt non accordé**'
        st.warning(score_text)
    else:
        score_text = 'La probabilité de défaillance du client est _très élevée_ ! **Prêt non accordé**'
        st.error(score_text)

st.divider()

if st.sidebar.checkbox('**Afficher les données du client**'):
    st.markdown(f"**Données du client n° {client_id}**")

    data_client = data[data["SK_ID_CURR"] == client_id]

    st.markdown(f"Age: {round(data_client['DAYS_BIRTH'].values[0] / -365)} ans")
    st.markdown(f"Est marié: {round(data_client['NAME_FAMILY_STATUS_Married'].values[0])}")
    st.markdown(f"A fait des études supérieures: {round(data_client['NAME_EDUCATION_TYPE_Higher_education'].values[0])}")
    st.markdown(f"Pourcentage de temps avec un emploi: {round(100 * data_client['DAYS_EMPLOYED_PERC'].values[0])} %")

    st.markdown(f"Montant du crédit demandé: {round(data_client['AMT_CREDIT'].values[0])}")
    st.markdown(f"Annuité : {data_client['AMT_ANNUITY'].values[0]}")
    st.markdown(f"Part de l'assurance crédit par rapport au crédit: {round(100 * data_client['PERC_INSURANCE_CRED'].values[0])} %")
    st.markdown(f"Nombre de paiements restant (autre crédit): {round(data_client['POS_REMAINING_INSTALMENTS'].values[0])}")

    temp_client = data_client[["SK_ID_CURR"] + ordered_features_list]
    temp_client["SK_ID_CURR"] = temp_client["SK_ID_CURR"].apply(lambda x: '{:,.0f}'.format(x).replace(',', ''))

    if st.checkbox('**Afficher les données complètes du client**'):
        st.dataframe(temp_client)
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
        nbr_var = st.slider("Nombre de variables", min_value=10, max_value=40, value=20, step=1)

        shap.decision_plot(explainer.expected_value, np.array(shap_v), data[data["SK_ID_CURR"] == client_id].drop(
            columns='SK_ID_CURR').iloc[0], link='logit', feature_display_range=slice(-1, -nbr_var - 1, -1))

        plt.gcf().set_size_inches(5, 6 * nbr_var / 20)

        # Display the plot in Streamlit
        st.pyplot(plt.gcf(), use_container_width=False)

        # Clear the current figure
        plt.clf()

        st.markdown(
            "Le diagramme de décision est un outil visuel qui permet d'expliquer les prédictions du modèle en "
            "montrant comment chaque variable influence la prédiction. "
            "<br>"
            "L'axe des abscisses représente la probabilité de défaillance du client. La ligne verticale grise indique "
            "la prédiction de défaillance moyenne de l'ensemble des clients. "
            "<br>"
            "L'axe des ordonnées liste les variables du modèle classées par ordre décroissant d'importance, "
            "en fonction de leur impact sur la prédiction. "
            "<br>"
            "Le diagramme démarre à la valeur de base (prédiction moyenne) et ajuste la prédiction en ajoutant la valeur SHAP de chaque variable.",
            unsafe_allow_html=True)

    if type_fig == 'Waterfall':
        nbr_var = st.slider("Nombre de variables", min_value=10, max_value=40, value=20, step=1)

        shap_values_instance = shap.Explanation(
            values=np.array(shap_v)[0],
            base_values=explainer.expected_value,
            data=data[data["SK_ID_CURR"] == client_id].drop(columns='SK_ID_CURR').iloc[0],
            feature_names=data.drop(columns='SK_ID_CURR').columns.tolist())

        # Plot the SHAP waterfall plot for a specific instance
        shap.waterfall_plot(shap_values_instance, max_display=nbr_var)

        plt.gcf().set_size_inches(5, 6 * nbr_var / 20)

        # Display the plot in Streamlit
        st.pyplot(plt.gcf(), use_container_width=False)

        # Clear the current figure
        plt.clf()

        st.markdown(
            "Le diagramme en cascade est un équivalent du diagramme de décision. La fonction logit est utilisée pour "
            "l'axe des abscisses.", unsafe_allow_html=True)

    st.divider()

if st.sidebar.checkbox('**Clients similaires**'):
    st.subheader("Données des 5 clients les plus similaires")

    nearest_clients = response['nearest_clients']
    st.dataframe(data.loc[data['SK_ID_CURR'].isin(nearest_clients)]
                 [["SK_ID_CURR"] + ordered_features_list].style.highlight_max(axis=0))

    average_probability = response['average_probability']
    average_probability = round(100 * average_probability)

    positive_cases = response['positive_cases']

    st.markdown(f"**Probabilité de défaillance moyenne des clients voisins:** {average_probability} %")
    st.markdown(f"**Nombre de clients voisins prédits comme défaillant:** {positive_cases}")

    if st.checkbox('**Afficher le decision plot**'):
        shap_v_nearest = response["shap_values_nearest"]

        shap.decision_plot(explainer.expected_value, np.array(shap_v_nearest),
                           data.loc[data['SK_ID_CURR'].isin(nearest_clients)].drop(columns='SK_ID_CURR'), link='logit',
                           highlight=data[data['SK_ID_CURR'] == 450148].index, plot_color="viridis")

        plt.gcf().set_size_inches(5, 6)

        # Display the plot in Streamlit
        st.pyplot(plt.gcf(), use_container_width=False)

        # Clear the current figure
        plt.clf()

    st.divider()
