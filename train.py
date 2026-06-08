# -*- coding: utf-8 -*-
"""
Entraînement et sérialisation du pipeline final (prétraitement + XGBoost).

Reprend exactement les étapes du notebook (nettoyage, encodage, normalisation,
tuning) et exporte `model_pipeline.joblib` utilisé par l'application Streamlit.

Usage :
    python train.py --data autos.csv
    python train.py                      # cherche autos.csv dans le dossier courant
"""
import argparse
import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CAT_COLS = ['vehicleType', 'gearbox', 'model', 'fuelType', 'brand', 'notRepairedDamage', 'abtest']
NUM_COLS = ['powerPS', 'kilometer', 'age']


def nettoyer(data):
    """Nettoyage identique au notebook, sans encodage ni normalisation."""
    data = data.copy()
    a_supprimer = ['index', 'lastSeen', 'dateCrawled', 'nrOfPictures', 'seller',
                   'offerType', 'postalCode', 'dateCreated', 'name', 'monthOfRegistration']
    data = data.drop(columns=[c for c in a_supprimer if c in data.columns])

    data = data[data['yearOfRegistration'].between(1980, 2023)]
    data['age'] = 2023 - data['yearOfRegistration']
    data = data.drop(columns=['yearOfRegistration'])

    data['gearbox'] = data['gearbox'].map({'manuell': 'Manual', 'automatik': 'Automatic'})
    data['notRepairedDamage'] = data['notRepairedDamage'].map({'ja': 'Yes', 'nein': 'No'})
    data['fuelType'] = data['fuelType'].replace({'benzin': 'Petrol', 'andere': 'Other'})

    data = data[data['price'].between(200, 20_000)]
    data = data[(data['powerPS'] > 0) & (data['powerPS'] <= 1000)]
    data = data[data['fuelType'] != 'Other']

    return data.dropna()


def main(chemin_csv):
    print(f'Lecture de {chemin_csv} ...')
    df = nettoyer(pd.read_csv(chemin_csv))
    print(f'Shape après nettoyage : {df.shape}')

    X = df[CAT_COLS + NUM_COLS]
    y = df['price']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=15)

    preprocesseur = ColumnTransformer([
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS),
    ])

    pipe = Pipeline([
        ('prep', preprocesseur),
        ('model', XGBRegressor(tree_method='hist', random_state=15)),
    ])

    grille = {
        'model__n_estimators': [400, 800, 1200],
        'model__max_depth': [4, 6, 8, 10],
        'model__learning_rate': [0.03, 0.05, 0.1],
        'model__subsample': [0.7, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.9, 1.0],
    }

    print('Tuning XGBoost (RandomizedSearchCV) ...')
    rs = RandomizedSearchCV(pipe, grille, n_iter=10, cv=3,
                            scoring='r2', n_jobs=-1, random_state=15, verbose=1)
    rs.fit(X_tr, y_tr)
    modele_final = rs.best_estimator_
    print(f'Meilleurs paramètres : {rs.best_params_}')

    pred_tr = modele_final.predict(X_tr)
    pred_te = modele_final.predict(X_te)
    r2_tr, r2_te = r2_score(y_tr, pred_tr), r2_score(y_te, pred_te)
    mae_te = mean_absolute_error(y_te, pred_te)
    rmse_te = mean_squared_error(y_te, pred_te) ** 0.5

    print('\n--- Modèle final : XGBoost ---')
    print(f'R2 train : {r2_tr:.4f}')
    print(f'R2 test  : {r2_te:.4f}  (écart train-test : {r2_tr - r2_te:.4f})')
    print(f'MAE test : {mae_te:.2f} €')
    print(f'RMSE test: {rmse_te:.2f} €')

    artefact = {
        'pipeline': modele_final,
        'cat_cols': CAT_COLS,
        'num_cols': NUM_COLS,
        'mae': float(mae_te),
        'r2_test': float(r2_te),
        'categories': {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS},
        'num_ranges': {c: {'min': float(df[c].min()),
                           'max': float(df[c].max()),
                           'median': float(df[c].median())} for c in NUM_COLS},
    }
    joblib.dump(artefact, 'model_pipeline.joblib')
    print('\nPipeline sérialisé dans model_pipeline.joblib ✅')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='autos.csv', help='Chemin vers le CSV autos.csv')
    args = parser.parse_args()
    main(args.data)
