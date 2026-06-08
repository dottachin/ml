# -*- coding: utf-8 -*-
"""
Application Streamlit — Estimation du prix d'une voiture d'occasion.

Charge le pipeline sérialisé (model_pipeline.joblib), laisse l'utilisateur
saisir les caractéristiques du véhicule, affiche les données saisies, la
prédiction et un niveau de confiance (intervalle prix ± MAE).
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Combien vaut votre voiture ?",
    page_icon="🔑",
    layout="centered",
)


@st.cache_resource
def charger_artefact(chemin='model_pipeline.joblib'):
    return joblib.load(chemin)


# ---------------------------------------------------------------------------
# Style — un parti pris éditorial, chaleureux et fait main (pas un formulaire
# générique). Papier crème, titres en serif, un seul accent terracotta.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&display=swap');

    :root {
        --paper:   #f7f3ec;
        --ink:     #2a2520;
        --muted:   #8a8073;
        --line:    #e4ddd0;
        --accent:  #c2521b;
        --accent-soft: #f0e4d8;
        --card:    #fffdf9;
    }

    .stApp { background: var(--paper); }

    /* Largeur de lecture confortable */
    .block-container { max-width: 760px; padding-top: 3rem; padding-bottom: 4rem; }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

    /* Cacher le chrome Streamlit pour un rendu "produit" */
    #MainMenu, footer, header { visibility: hidden; }

    /* En-tête */
    .kicker {
        font-size: .78rem; letter-spacing: .18em; text-transform: uppercase;
        color: var(--accent); font-weight: 600; margin-bottom: .6rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif; font-weight: 600;
        font-size: 2.9rem; line-height: 1.05; letter-spacing: -.01em;
        margin: 0 0 1rem 0; color: var(--ink);
    }
    .hero-title em { font-style: italic; color: var(--accent); }
    .hero-sub {
        font-size: 1.06rem; line-height: 1.6; color: var(--muted);
        max-width: 56ch; margin-bottom: .4rem;
    }

    /* Étiquettes de section */
    .section-label {
        font-family: 'Fraunces', serif; font-size: 1.25rem; font-weight: 600;
        margin: 2.4rem 0 .2rem 0;
    }
    .section-hint { color: var(--muted); font-size: .92rem; margin-bottom: 1rem; }

    /* Selectbox & sliders — adoucir les angles, fond carte */
    div[data-baseweb="select"] > div {
        background: var(--card) !important;
        border-radius: 12px !important;
        border-color: var(--line) !important;
    }
    .stSlider label, .stSelectbox label {
        font-weight: 500 !important; color: var(--ink) !important; font-size: .9rem !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }

    /* Bouton principal */
    .stButton > button {
        background: var(--accent); color: #fff; border: none;
        border-radius: 999px; padding: .8rem 2rem; font-weight: 600;
        font-size: 1rem; width: 100%; transition: transform .12s ease, box-shadow .12s ease;
        box-shadow: 0 6px 20px -8px rgba(194,82,27,.6);
    }
    .stButton > button:hover {
        transform: translateY(-1px); color: #fff;
        box-shadow: 0 10px 26px -8px rgba(194,82,27,.7);
    }

    /* Carte résultat */
    .result-card {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 22px; padding: 2.2rem 2.2rem 2rem; margin-top: 1.6rem;
        box-shadow: 0 24px 60px -34px rgba(42,37,32,.35);
    }
    .result-eyebrow { color: var(--muted); font-size: .9rem; margin-bottom: .3rem; }
    .result-price {
        font-family: 'Fraunces', serif; font-size: 3.4rem; font-weight: 600;
        line-height: 1; color: var(--ink); margin-bottom: .2rem;
    }
    .result-price span { font-size: 1.6rem; color: var(--muted); }

    /* Barre de fourchette */
    .range-track {
        height: 10px; border-radius: 999px; margin: 1.6rem 0 .6rem;
        background: linear-gradient(90deg, var(--accent-soft), var(--accent), var(--accent-soft));
        position: relative;
    }
    .range-dot {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
        width: 20px; height: 20px; border-radius: 50%; background: #fff;
        border: 4px solid var(--accent); box-shadow: 0 3px 8px rgba(0,0,0,.18);
    }
    .range-ends { display: flex; justify-content: space-between; color: var(--muted); font-size: .9rem; }
    .range-ends b { color: var(--ink); font-weight: 600; }

    .reassure {
        margin-top: 1.3rem; padding-top: 1.3rem; border-top: 1px dashed var(--line);
        color: var(--muted); font-size: .9rem; line-height: 1.55;
    }

    /* Bandeau de confiance modèle */
    .trust { display: flex; gap: 1.8rem; flex-wrap: wrap; margin-top: 1.4rem; }
    .trust-item { display: flex; flex-direction: column; }
    .trust-num { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600; color: var(--ink); }
    .trust-lbl { color: var(--muted); font-size: .82rem; }

    .footer-note { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    art = charger_artefact()
except FileNotFoundError:
    st.error("Fichier `model_pipeline.joblib` introuvable. "
             "Lance d'abord `python train.py --data autos.csv` pour le générer.")
    st.stop()

pipeline = art['pipeline']
cat_cols = art['cat_cols']
num_cols = art['num_cols']
categories = art['categories']
ranges = art['num_ranges']
mae = art['mae']

# --- En-tête ---
st.markdown('<div class="kicker">Estimation gratuite · sans engagement</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">Combien vaut <em>vraiment</em><br>votre voiture&nbsp;?</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">Donnez-nous quelques détails sur votre véhicule — la marque, '
    "l'année, les kilomètres au compteur — et nous vous dirons à combien il se vend "
    "aujourd'hui sur le marché de l'occasion. Ça prend une minute.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="trust">
      <div class="trust-item">
        <span class="trust-num">{art['r2_test']:.0%}</span>
        <span class="trust-lbl">de fiabilité (R² sur données test)</span>
      </div>
      <div class="trust-item">
        <span class="trust-num">± {mae:,.0f} €</span>
        <span class="trust-lbl">marge d'erreur moyenne</span>
      </div>
    </div>
    """.replace(",", " "),
    unsafe_allow_html=True,
)

# --- Étiquettes lisibles pour les colonnes ---
labels = {
    'vehicleType': "Type de véhicule",
    'gearbox': "Boîte de vitesses",
    'model': "Modèle",
    'fuelType': "Carburant",
    'brand': "Marque",
    'notRepairedDamage': "Dommage non réparé",
    'abtest': "Groupe A/B",
    'powerPS': "Puissance (ch)",
    'kilometer': "Kilométrage (km)",
    'age': "Âge du véhicule (années)",
}

# --- Formulaire de saisie ---
st.markdown('<div class="section-label">Parlez-nous du véhicule</div>', unsafe_allow_html=True)
st.markdown('<div class="section-hint">Les essentiels pour le reconnaître.</div>', unsafe_allow_html=True)

saisie = {}
col1, col2 = st.columns(2)

with col1:
    saisie['brand'] = st.selectbox(labels['brand'], categories['brand'])
    saisie['model'] = st.selectbox(labels['model'], categories['model'])
    saisie['vehicleType'] = st.selectbox(labels['vehicleType'], categories['vehicleType'])
    saisie['gearbox'] = st.selectbox(labels['gearbox'], categories['gearbox'])

with col2:
    saisie['fuelType'] = st.selectbox(labels['fuelType'], categories['fuelType'])
    saisie['notRepairedDamage'] = st.selectbox(labels['notRepairedDamage'], categories['notRepairedDamage'])
    saisie['abtest'] = st.selectbox(labels['abtest'], categories['abtest'])

st.markdown('<div class="section-label">L\'état et l\'usage</div>', unsafe_allow_html=True)
st.markdown('<div class="section-hint">Ce qui fait grimper — ou baisser — la cote.</div>', unsafe_allow_html=True)

for c in num_cols:
    r = ranges[c]
    saisie[c] = st.slider(labels[c], int(r['min']), int(r['max']), int(r['median']))

# --- Construction de la ligne d'entrée (mêmes colonnes que l'entraînement) ---
entree = pd.DataFrame([saisie])[cat_cols + num_cols]

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# --- Prédiction ---
if st.button("Estimer mon prix", type="primary"):
    prix = float(pipeline.predict(entree)[0])
    borne_basse = max(0, prix - mae)
    borne_haute = prix + mae

    # Position du repère sur la barre (le prix estimé au sein de sa fourchette)
    largeur = borne_haute - borne_basse
    pos = 50 if largeur == 0 else round((prix - borne_basse) / largeur * 100)

    def euro(v):
        return f"{v:,.0f} €".replace(",", " ")

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-eyebrow">Votre {saisie['brand'].title()} {str(saisie['model']).title()} se vendrait autour de</div>
          <div class="result-price">{euro(prix)}</div>

          <div class="range-track">
            <div class="range-dot" style="left:{pos}%"></div>
          </div>
          <div class="range-ends">
            <span><b>{euro(borne_basse)}</b><br>au plus bas</span>
            <span style="text-align:right"><b>{euro(borne_haute)}</b><br>au plus haut</span>
          </div>

          <div class="reassure">
            La plupart des annonces similaires tombent dans cette fourchette. L'écart
            correspond à l'erreur moyenne du modèle (± {euro(mae)}) — un véhicule très bien
            entretenu vise le haut, un modèle fatigué le bas.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Voir les détails que vous avez saisis"):
        st.dataframe(entree, use_container_width=True)

st.markdown(
    '<div class="footer-note">Estimation indicative basée sur un modèle XGBoost '
    "entraîné sur des milliers d'annonces réelles. Le prix final dépend de l'état "
    "exact, des options et de la négociation.</div>",
    unsafe_allow_html=True,
)
