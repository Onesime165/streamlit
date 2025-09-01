import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from plotly.subplots import make_subplots
import matplotlib.font_manager as fm

# =================================================================================================
# Configuration de la page Streamlit
# =================================================================================================
st.set_page_config(
    page_title="Onesime Vision X | Analyse Visuelle de Données",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================================================
# CSS Personnalisé pour un look premium amélioré
# =================================================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman:ital,wght@0,400;0,700;1,400&display=swap');
    
    /* Thème général amélioré avec gradient plus subtil */
    .stApp {
        background: linear-gradient(135deg, #0a001f 0%, #14003a 100%);
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    
    /* Titres avec effet glow subtil */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Times New Roman', serif !important;
        font-weight: 700;
        color: #d4bfff;
        text-shadow: 0 0 8px rgba(212, 191, 255, 0.3);
        letter-spacing: 0.6px;
    }
    
    /* Texte standard */
    .stMarkdown, .stText, .stDataFrame, div {
        font-family: 'Times New Roman', serif !important;
    }
    
    /* Barre latérale améliorée avec glassmorphism */
    .css-1d391kg {
        background: rgba(15, 5, 40, 0.9);
        backdrop-filter: blur(12px);
        border-right: 1px solid #7a4dff;
        box-shadow: 4px 0 20px rgba(122, 77, 255, 0.25);
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Widgets Streamlit avec transitions plus fluides */
    .stButton>button {
        background: linear-gradient(135deg, #6e00cc 0%, #5200e6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-family: 'Times New Roman', serif;
        font-weight: 700;
        transition: all 0.4s ease;
        box-shadow: 0 4px 8px rgba(110, 0, 204, 0.25);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #7f1fff 0%, #6320ff 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(110, 0, 204, 0.35);
    }
    
    .stSelectbox, .stFileUploader, .stNumberInput, .stTextInput {
        border-radius: 10px;
        border: 1px solid #3e1f80;
        background: rgba(25, 15, 50, 0.8);
        color: white;
    }
    
    /* Conteneurs de graphiques avec glassmorphism */
    .block-container {
        padding: 2.5rem 1.5rem;
    }
    
    .stPlotlyChart, .stImage, .stPydeckChart {
        border-radius: 15px;
        padding: 1.2rem;
        background: rgba(255, 255, 255, 0.03);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(122, 77, 255, 0.15);
        margin-bottom: 2rem;
    }
    
    /* Dataframes avec bordures arrondies */
    .stDataFrame {
        border-radius: 10px;
        background: rgba(25, 15, 50, 0.6);
        border: 1px solid rgba(122, 77, 255, 0.15);
    }
    
    /* Séparateurs stylisés */
    hr {
        border-color: #3e1f80;
        margin: 2.5rem 0;
        border-style: dashed;
        border-width: 1px 0 0 0;
    }
    
    /* Cards avec depth améliorée */
    .card {
        background: rgba(25, 15, 50, 0.6);
        border-radius: 15px;
        padding: 1.8rem;
        border: 1px solid rgba(122, 77, 255, 0.15);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }
    
    /* Metrics avec fonts plus grandes */
    [data-testid="stMetricValue"] {
        font-family: 'Times New Roman', serif;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =================================================================================================
# Fonctions Utilitaires
# =================================================================================================

@st.cache_data
def create_sample_data():
    """Génère un DataFrame d'exemple avec des données variées."""
    np.random.seed(42)
    data = {
        'ID_Employe': range(1001, 1101),
        'Departement': np.random.choice(['Ingénierie', 'Ressources Humaines', 'Marketing', 'Ventes', 'Finance'], 100),
        'Age': np.random.randint(22, 60, 100),
        'Salaire_Annuel_K': np.random.normal(loc=80, scale=20, size=100).round(1),
        'Satisfaction_Employe': np.random.randint(1, 6, 100),
        'Ville': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Bordeaux', 'Nantes', 'Strasbourg'], 100),
        'Anciennete': np.random.randint(1, 15, 100),
        'Performance': np.random.normal(loc=75, scale=15, size=100).round(1)
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

def display_data_summary(df, selected_column):
    """Affiche un résumé statistique de la colonne sélectionnée avec layout équilibré."""
    st.markdown(f"### 📋 Résumé Statistique: `{selected_column}`")
    
    if is_quantitative(df[selected_column]):
        cols = st.columns(4)
        metrics = [
            ("Moyenne", f"{df[selected_column].mean():.2f}"),
            ("Médiane", f"{df[selected_column].median():.2f}"),
            ("Écart-type", f"{df[selected_column].std():.2f}"),
            ("Valeurs Uniques", df[selected_column].nunique())
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
    else:
        cols = st.columns(3)
        metrics = [
            ("Valeurs Uniques", df[selected_column].nunique()),
            ("Valeur la plus fréquente", df[selected_column].mode().iloc[0] if not df[selected_column].mode().empty else "N/A"),
            ("Count", f"{df[selected_column].count()}")
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
    
    st.markdown("---")

# =================================================================================================
# Fonctions de Visualisation
# =================================================================================================

def display_quantitative_visuals(df, column):
    """Affiche 4 visualisations pour une variable quantitative avec layout équilibré."""
    st.markdown(f'<div class="card"><h3>📈 Analyse Quantitative: {column}</h3></div>', unsafe_allow_html=True)
    
    # Statistiques descriptives
    display_data_summary(df, column)
    
    # Layout en 2x2 équilibré avec containers pour uniformiser les hauteurs
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1.container():
        # 1. Histogramme avec courbe de densité
        st.markdown("##### 📊 Distribution (Histogramme)")
        fig_hist = px.histogram(df, x=column, marginal="box", nbins=30,
                                color_discrete_sequence=['#8a54ff'],
                                template="plotly_dark",
                                title=f"Distribution de {column}")
        fig_hist.update_layout(
            font_family="Times New Roman",
            title_font_size=18,
            bargap=0.1,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with row1_col2.container():
        # 2. Boîte à Moustaches (Box Plot)
        st.markdown("##### 📦 Synthèse Statistique (Box Plot)")
        fig_box = px.box(df, y=column,
                         color_discrete_sequence=['#00c8ff'],
                         template="plotly_dark",
                         title=f"Box Plot de {column}")
        fig_box.update_layout(
            font_family="Times New Roman",
            title_font_size=18,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with row2_col1.container():
        # 3. Diagramme de densité (KDE)
        st.markdown("##### 📈 Courbe de Densité (KDE)")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
        sns.kdeplot(data=df, x=column, fill=True, color="#8a54ff", ax=ax, alpha=0.7)
        ax.set_title(f'Distribution de la densité de {column}', fontfamily='Times New Roman', color='white', fontsize=14)
        ax.set_xlabel(column, fontfamily='Times New Roman', color='white')
        ax.set_ylabel('Densité', fontfamily='Times New Roman', color='white')
        ax.tick_params(colors='white')
        ax.grid(alpha=0.2)
        fig.set_facecolor('none')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', transparent=True, dpi=120)
        st.image(buf)

    with row2_col2.container():
        # 4. Graphique en violon
        st.markdown("##### 🎻 Densité et Répartition (Violon)")
        fig_violin = px.violin(df, y=column, box=True, points="all",
                               color_discrete_sequence=['#c792ea'],
                               template="plotly_dark",
                               title=f"Densité de {column}")
        fig_violin.update_layout(
            font_family="Times New Roman",
            title_font_size=18,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_violin, use_container_width=True)

def display_qualitative_visuals(df, column):
    """Affiche 4 visualisations pour une variable qualitative avec layout équilibré."""
    st.markdown(f'<div class="card"><h3>📊 Analyse Qualitative: {column}</h3></div>', unsafe_allow_html=True)
    
    # Statistiques descriptives
    display_data_summary(df, column)
    
    # Layout en 2x2 équilibré avec containers pour uniformiser les hauteurs
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1.container():
        # 1. Diagramme en barres
        st.markdown("##### 📊 Fréquence des Catégories (Barres)")
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, 'count']
        fig_bar = px.bar(counts, x=column, y='count',
                         color=column,
                         template="plotly_dark",
                         title=f"Nombre d'occurrences par catégorie",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_bar.update_layout(
            font_family="Times New Roman",
            title_font_size=18,
            height=450,
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with row1_col2.container():
        # 2. Treemap
        st.markdown("##### 🌳 Proportion des Catégories (Treemap)")
        fig_treemap = px.treemap(df, path=[px.Constant("Toutes les catégories"), column],
                                 template="plotly_dark",
                                 title=f"Treemap de {column}",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_treemap.update_layout(
            font_family="Times New Roman",
            title_font_size=18,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
        
    with row2_col1.container():
        # 3. Diagramme circulaire (Pie Chart)
        st.markdown("##### 🥧 Répartition en Pourcentage (Camembert)")
        fig_pie = px.pie(df, names=column,
                         hole=0.4,
                         template="plotly_dark",
                         title=f"Répartition de {column}",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(
            font_family="Times New Roman",
            title_font_size=18,
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with row2_col2.container():
        # 4. Nuage de Mots
        st.markdown("##### ☁️ Nuage de Mots")
        text = ' '.join(df[column].dropna().astype(str).tolist())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA',
                                  colormap='plasma', max_font_size=150, min_font_size=10).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Données insuffisantes pour générer un nuage de mots.")

# =================================================================================================
# Interface Principale de l'Application
# =================================================================================================

# En-tête avec style amélioré et animation subtile
st.markdown("""
<div style="text-align: center; padding: 2.5rem 0; background: linear-gradient(135deg, rgba(110, 0, 204, 0.25) 0%, rgba(25, 12, 55, 0.45) 100%); border-radius: 15px; margin-bottom: 2.5rem; box-shadow: 0 4px 15px rgba(110, 0, 204, 0.2);">
    <h1 style="font-size: 3rem; margin-bottom: 0.6rem;">💡 VisionX</h1>
    <h3 style="font-weight: 400; margin-top: 0;">Explorateur Visuel de Données Avancé</h3>
    <p style="opacity: 0.85; font-size: 1.1rem;">Une application intelligente pour générer des visualisations percutantes à partir de vos données</p>
</div>
""", unsafe_allow_html=True)

# --- Barre latérale ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem;">
        <h2>⚙️ Centre de Contrôle</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Chargez votre fichier de données",
        type=['csv', 'xlsx', 'xls'],
        help="Formats supportés: CSV, Excel"
    )

    df_sample = create_sample_data()
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("Fichier chargé avec succès!")
    else:
        st.info("Aucun fichier chargé. Utilisation d'un jeu de données d'exemple.")
        df = df_sample
        
    st.markdown("---")
    
    # Options d'analyse
    if 'df' in locals() and df is not None:
        st.markdown("### 🔍 Options d'Analyse")
        
        if not df.columns.empty:
            selected_column = st.selectbox(
                "Choisissez une variable à analyser :",
                df.columns,
                help="Sélectionnez une colonne pour générer des visualisations"
            )
        else:
            st.warning("Le jeu de données est vide ou n'a pas de colonnes.")
            selected_column = None
            
        st.markdown("---")
        
        # Informations sur le dataset
        st.markdown("### 📊 Informations du Dataset")
        cols = st.columns(3)
        cols[0].write(f"**Lignes:** {df.shape[0]}")
        cols[1].write(f"**Colonnes:** {df.shape[1]}")
        cols[2].write(f"**Valeurs manquantes:** {df.isnull().sum().sum()}")
        
        # Aperçu des types de données
        if st.checkbox("Afficher les types de données"):
            st.write(df.dtypes)
            
        st.markdown("---")
        st.markdown("*Développé avec ❤️ utilisant Streamlit*")

# --- Zone principale ---
if 'df' in locals() and df is not None:
    # Aperçu des données avec style amélioré
    st.markdown("### 📋 Aperçu des Données")
    
    expander = st.expander("Explorer le jeu de données", expanded=False)
    with expander:
        cols = st.columns(3)
        cols[0].metric("Nombre de lignes", df.shape[0])
        cols[1].metric("Nombre de colonnes", df.shape[1])
        cols[2].metric("Valeurs manquantes", df.isnull().sum().sum())
        
        st.dataframe(df.head(10), use_container_width=True)
    
    # Analyse de la colonne sélectionnée
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

