"""Limpieza y feature engineering, portados desde prediction_movies_imdb.ipynb
(repo aprendizaje_de_maquinas_tp): duplicados, películas adultas, runtime/budget/
revenue inválidos, encoding de original_language/genres/production_countries/
production_companies.
"""
from sklearn.model_selection import train_test_split

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


def split(df):
    # Target
    df["profitable"] = (df["revenue"] >= df["budget"]).astype(int)

    # Features que sí queremos conservar
    X = df.drop(columns=["profitable","revenue","id","adult", "popularity", "vote_average", "vote_count"])
    y = df["profitable"]
    X.info()
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=14,
        stratify=y
    )
    return X_train, X_test, y_train, y_test