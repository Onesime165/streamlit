import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

# =================================================================================================
# Configuration de la page Streamlit
# =================================================================================================
st.set_page_config(
    page_title="VisionX | Analyse Visuelle",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================================================
# CSS Personnalisé pour un look technologique et naturel
# =================================================================================================
st.markdown("""
<style>
    /* Thème général */
    .stApp {
        background-color: #0c002b; /* Fond bleu nuit profond */
        color: #e0e0e0; /* Texte gris clair */
    }

    /* Barre latérale */
    .css-1d391kg {
        background-color: rgba(38, 39, 48, 0.4); /* Fond semi-transparent */
        border-right: 1px solid #7a00e0; /* Bordure violet néon */
    }

    /* Titres */
    h1, h2, h3 {
        color: #c792ea; /* Violet pastel pour les titres */
        text-shadow: 2px 2px 5px #000000;
    }

    /* Widgets Streamlit */
    .stButton>button {
        color: #ffffff;
        background-color: #7a00e0; /* Violet néon */
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #c792ea; /* Violet plus clair au survol */
        box-shadow: 0 0 15px #c792ea;
    }
    .stSelectbox, .stFileUploader {
        border-radius: 10px;
    }

    /* Conteneurs de graphiques */
    .stPlotlyChart, .stImage, .stPydeckChart {
        border-radius: 15px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.05); /* Fond très léger pour les graphiques */
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(122, 0, 224, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# =================================================================================================
# Fonctions Utilitaires
# =================================================================================================

@st.cache_data
def create_sample_data():
    """Génère un DataFrame d'exemple avec des données variées."""
    data = {
        'ID_Employe': range(1001, 1101),
        'Departement': np.random.choice(['Ingénierie', 'Ressources Humaines', 'Marketing', 'Ventes', 'Finance'], 100),
        'Age': np.random.randint(22, 60, 100),
        'Salaire_Annuel_K': np.random.normal(loc=80, scale=20, size=100).round(1),
        'Satisfaction_Employe': np.random.randint(1, 6, 100),
        'Ville': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'], 100)
    }
    return pd.DataFrame(data)

@st.cache_data
def load_data(uploaded_file):
    """Charge les données depuis un fichier CSV ou Excel."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return None

def is_quantitative(series):
    """Détermine si une variable est quantitative."""
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10

def is_qualitative(series):
    """Détermine si une variable est qualitative."""
    return pd.api.types.is_object_dtype(series) or series.nunique() <= 10


# =================================================================================================
# Fonctions de Visualisation
# =================================================================================================

def display_quantitative_visuals(df, column):
    """Affiche 4 visualisations pour une variable quantitative."""
    st.subheader(f"Analyse Quantitative de : `{column}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Histogramme
        st.write("#### 📊 Distribution (Histogramme)")
        fig_hist = px.histogram(df, x=column, marginal="box",
                                color_discrete_sequence=['#7a00e0'],
                                template="plotly_dark",
                                title=f"Distribution de {column}")
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

        # 2. Graphique en violon
        st.write("#### 🎻 Densité et Répartition (Violon)")
        fig_violin = px.violin(df, y=column, box=True, points="all",
                               color_discrete_sequence=['#c792ea'],
                               template="plotly_dark",
                               title=f"Densité de {column}")
        st.plotly_chart(fig_violin, use_container_width=True)

    with col2:
        # 3. Boîte à Moustaches (Box Plot)
        st.write("#### 📦 Synthèse Statistique (Box Plot)")
        fig_box = px.box(df, y=column,
                         color_discrete_sequence=['#00aaff'],
                         template="plotly_dark",
                         title=f"Box Plot de {column}")
        st.plotly_chart(fig_box, use_container_width=True)
        
        # 4. Diagramme de densité (KDE)
        st.write("#### 📈 Courbe de Densité (KDE)")
        fig, ax = plt.subplots(facecolor='#0c002b')
        sns.kdeplot(data=df, x=column, fill=True, color="#7a00e0", ax=ax)
        ax.set_title(f'Distribution de la densité de {column}', color='white')
        ax.set_xlabel(column, color='white')
        ax.set_ylabel('Densité', color='white')
        ax.tick_params(colors='white')
        ax.grid(alpha=0.2)
        fig.set_facecolor('#0c002b')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
        st.image(buf)


def display_qualitative_visuals(df, column):
    """Affiche 4 visualisations pour une variable qualitative."""
    st.subheader(f"Analyse Qualitative de : `{column}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Diagramme en barres
        st.write("#### 📊 Fréquence des Catégories (Barres)")
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'count']
        fig_bar = px.bar(counts, x=column, y='count',
                         color=column,
                         template="plotly_dark",
                         title=f"Nombre d'occurrences par catégorie")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 2. Treemap
        st.write("#### 🌳 Proportion des Catégories (Treemap)")
        fig_treemap = px.treemap(df, path=[px.Constant("Toutes les catégories"), column],
                                 template="plotly_dark",
                                 title=f"Treemap de {column}")
        fig_treemap.update_traces(root_color="lightgrey")
        st.plotly_chart(fig_treemap, use_container_width=True)

    with col2:
        # 3. Diagramme circulaire (Pie Chart)
        st.write("#### 🥧 Répartition en Pourcentage (Camembert)")
        fig_pie = px.pie(df, names=column,
                         hole=0.4,
                         template="plotly_dark",
                         title=f"Répartition de {column}")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 4. Nuage de Mots
        st.write("#### ☁️ Nuage de Mots")
        text = ' '.join(df[column].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA',
                              colormap='viridis').generate(text)
        fig, ax = plt.subplots(facecolor='none')
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        

# =================================================================================================
# Interface Principale de l'Application
# =================================================================================================

st.title("💡 VisionX : Explorateur Visuel de Données")
st.markdown("##### Une application intelligente pour générer des visualisations percutantes à partir de vos données.")

# --- Barre latérale ---
with st.sidebar:
    st.header("⚙️ Centre de Contrôle")
    uploaded_file = st.file_uploader(
        "Chargez votre fichier (CSV ou Excel)",
        type=['csv', 'xlsx', 'xls']
    )

    df_sample = create_sample_data()
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.info("Aucun fichier chargé. Utilisation d'un jeu de données d'exemple.")
        df = df_sample
        
# --- Zone principale ---
if 'df' in locals() and df is not None:
    st.write("### Aperçu des Données")
    st.dataframe(df.head(), use_container_width=True)

    with st.sidebar:
        if not df.columns.empty:
            selected_column = st.selectbox(
                "Choisissez une variable à analyser :",
                df.columns
            )
        else:
            st.warning("Le jeu de données est vide ou n'a pas de colonnes.")
            selected_column = None
    
    if selected_column:
        st.markdown("---")
        
        # Détection du type de variable et affichage des graphiques correspondants
        if is_quantitative(df[selected_column]):
            display_quantitative_visuals(df, selected_column)
        elif is_qualitative(df[selected_column]):
            display_qualitative_visuals(df, selected_column)
        else:
            st.warning(f"La colonne `{selected_column}` n'est ni clairement quantitative ni qualitative. Veuillez choisir une autre colonne.")

else:
    st.info("Veuillez charger un fichier ou utiliser les données d'exemple pour commencer.")