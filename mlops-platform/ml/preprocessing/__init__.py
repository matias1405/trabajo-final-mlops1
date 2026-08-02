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
    from sklearn.preprocessing import MultiLabelBinarizer

    # convertir variables categóricas separadas por coma a listas
    for col in ["genres", "production_countries", "production_companies"]:
        df[col] = df[col].fillna("").apply(
            lambda x: [v.strip() for v in x.split(",") if v.strip()]
        )

    # cantidad de géneros por película
    df["n_genres"] = df["genres"].apply(len)

    # original_language: conservar top idiomas y agrupar el resto
    top_languages = df["original_language"].value_counts().nlargest(10).index

    df.loc[~df["original_language"].isin(top_languages), "original_language"] = "other"

    language_encoded = pd.get_dummies(
        df["original_language"],
        prefix="lang",
        dtype=int
    )

    df = pd.concat(
        [df.drop(columns=["original_language"]), language_encoded],
        axis=1
    )
    
    # genres: multi-hot encoding
    mlb_genres = MultiLabelBinarizer()

    genres_encoded = mlb_genres.fit_transform(df["genres"])

    genres_df = pd.DataFrame(
        genres_encoded,
        columns=[f"genre_{g}" for g in mlb_genres.classes_],
        index=df.index
    )

    df = pd.concat(
        [df.drop(columns=["genres"]), genres_df],
        axis=1
    )

    # production_countries: multi-hot encoding top 25
    top_countries = (
        df["production_countries"]
        .explode()
        .value_counts()
        .nlargest(25)
        .index
    )

    mlb_countries = MultiLabelBinarizer(classes=top_countries.tolist())

    countries_encoded = mlb_countries.fit_transform(
        df["production_countries"].apply(
            lambda x: [c for c in x if c in top_countries]
        )
    )

    countries_df = pd.DataFrame(
        countries_encoded,
        columns=[f"country_{c}" for c in mlb_countries.classes_],
        index=df.index
    )

    df = pd.concat(
        [df.drop(columns=["production_countries"]), countries_df],
        axis=1
    )
    
    # production_companies: multi-hot encoding top 35
    top_companies = (
        df["production_companies"]
        .explode()
        .value_counts()
        .nlargest(35)
        .index
    )

    mlb_companies = MultiLabelBinarizer(classes=top_companies.tolist())

    companies_encoded = mlb_companies.fit_transform(
        df["production_companies"].apply(
            lambda x: [c for c in x if c in top_companies]
        )
    )

    companies_df = pd.DataFrame(
        companies_encoded,
        columns=[f"company_{c}" for c in mlb_companies.classes_],
        index=df.index
    )

    df = pd.concat(
        [df.drop(columns=["production_companies"]), companies_df],
        axis=1
    )
    
    # transformar budget y revenue a escala logarítmica
    df["budget"] = np.log1p(df["budget"])
    df["revenue"] = np.log1p(df["revenue"])

    # crear variable target
    df["profitable"] = (df["revenue"] >= df["budget"]).astype(int)

    # extraer mes de lanzamiento
    df["release_month"] = pd.to_datetime(df["release_date"]).dt.month

    # eliminar revenue para evitar data leakage
    df = df.drop(columns=["revenue"])

    return df