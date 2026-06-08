# 🚗 Prédiction du prix des voitures d'occasion

Projet ML de bout en bout : de la compréhension du problème métier jusqu'au déploiement
d'une application interactive de prédiction.

## 📌 Description du projet

À partir d'annonces de voitures d'occasion (jeu de données eBay Allemagne), on estime
automatiquement le **prix** d'un véhicule à partir de ses caractéristiques (marque, modèle,
kilométrage, puissance, âge, boîte de vitesses, carburant, état…).

Le prix étant une grandeur continue, il s'agit d'une tâche de **régression supervisée**.

Le projet couvre l'ensemble du pipeline ML :

1. **Compréhension de la problématique** + état de l'art
2. **Analyse exploratoire (EDA)** : types de variables, cible, corrélations, distributions
3. **Prétraitement** : valeurs manquantes, traduction des catégories, outliers, encodage, normalisation
4. **Modélisation** : un DNN + plusieurs modèles de régression (≥ 10 algorithmes testés)
5. **Explicabilité (XAI)** avec SHAP
6. **Tuning** des 3 modèles candidats (XGBoost, CatBoost, Random Forest) + choix du modèle final
7. **Déploiement** : pipeline sérialisé (`joblib`) + application Streamlit


## 📊 Modèle final & résultats

| Élément | Valeur |
|---|---|
| Modèle final | **XGBoost** (pipeline scikit-learn) |
| R² (test) | ~0.90 |
| MAE (test) | erreur moyenne en € (affichée dans l'app) |
| Sur-apprentissage | écart train/test faible → généralisation correcte |

Variables les plus déterminantes (SHAP) : **`age`** et **`powerPS`**.

## 🗂️ Structure du dépôt

```
.
├── app.py                                          # Application Streamlit (prédiction)
├── train.py                                        # Entraînement + sérialisation du pipeline
├── model_pipeline.joblib                           # Pipeline complet sérialisé (généré par train.py)
├── used-cars-eda-dnn-ml-xai-ensemble-r2-90.ipynb   # Notebook complet (EDA, modèles, tuning)
├── requirements.txt                                # Dépendances de l'application
├── rapport.pdf                                      # Rapport du projet
└── README.md
```

## ▶️ Reproduire le projet

### 1. Générer le pipeline sérialisé
Le fichier `model_pipeline.joblib` est produit à partir du jeu de données `autos.csv`
([Kaggle](https://www.kaggle.com/datasets/orgesleka/used-cars-database) / *Uncovering Factors That Affect Used Car Prices*) :

```bash
pip install -r requirements.txt
python train.py --data autos.csv
```

> Le notebook produit également ce fichier dans sa dernière section (cellule de sérialisation).

### 2. Lancer l'application en local

```bash
streamlit run app.py
```