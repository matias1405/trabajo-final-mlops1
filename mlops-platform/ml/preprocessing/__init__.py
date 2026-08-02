"""Limpieza y feature engineering, portados desde prediction_movies_imdb.ipynb
(repo aprendizaje_de_maquinas_tp): duplicados, películas adultas, runtime/budget/
revenue inválidos, encoding de original_language/genres/production_countries/
production_companies.
"""


def validate(df):
    required_columns = [
        "id",
        "adult",
        "runtime",
        "budget",
        "revenue",
        "release_date",
        "original_language",
        "genres",
        "production_countries",
        "production_companies",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    if df.empty:
        raise ValueError("El dataset está vacío")

    return True


def clean(df):
    # eliminar duplicados
    df = df.drop_duplicates(subset="id")

    # eliminar películas adultas
    df = df[df["adult"] == False]

    # eliminar datos inválidos
    df = df[df["runtime"] > 0]
    df = df[df["budget"] > 0]
    df = df[df["revenue"] > 0]

    # eliminar películas sin fecha de lanzamiento
    df = df[df["release_date"].notna()]

    return df


def engineer_features(df):
    import numpy as np
    import pandas as pd

    # transformar budget y revenue a escala logarítmica
    df["budget"] = np.log1p(df["budget"])
    df["revenue"] = np.log1p(df["revenue"])

    # extraer mes de lanzamiento
    df["release_month"] = pd.to_datetime(df["release_date"]).dt.month

    # crear variable target
    df["profitable"] = (df["revenue"] >= df["budget"]).astype(int)

    # eliminar revenue para evitar data leakage
    df = df.drop(columns=["revenue"])

    return df