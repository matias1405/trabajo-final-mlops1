"""Limpieza y feature engineering, portados desde prediction_movies_imdb.ipynb
(repo aprendizaje_de_maquinas_tp): duplicados, películas adultas, runtime/budget/
revenue inválidos, encoding de original_language/genres/production_countries/
production_companies.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


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


def _as_list(value):
    # Acepta tanto listas ya parseadas (requests de inferencia) como strings
    # separadas por coma (columnas crudas del CSV de entrenamiento).
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _prepare_base_features(df):
    """Transformaciones fila a fila: no dependen de estadísticas del dataset
    completo (a diferencia del top-N de idiomas/países/productoras, que sí
    debe fitearse solo sobre train — ver FeatureEncoder), así que da lo mismo
    aplicarlas antes o después del split.
    """
    df = df.copy()

    for col in ["genres", "production_countries", "production_companies"]:
        df[col] = df[col].apply(_as_list)

    df["n_genres"] = df["genres"].apply(len)

    if "release_date" in df.columns:
        df["release_month"] = pd.to_datetime(df["release_date"]).dt.month

    return df


@dataclass
class FeatureEncoder:
    """Encoding fiteado únicamente sobre el split de entrenamiento: idiomas
    top-10 y los MultiLabelBinarizer de géneros y de países/productoras
    top-25/top-35. Se persiste como artefacto de MLflow junto con el modelo
    (ver ml.training.register_model) para que ml/inference reproduzca
    exactamente el mismo encoding sobre requests nuevos, en vez de tener que
    adivinar columnas a partir de `model.feature_names_in_` o recalcular
    top-N sobre datos que el modelo nunca debería ver (el propio test set,
    o requests de producción).
    """

    top_languages: list
    mlb_genres: MultiLabelBinarizer
    mlb_countries: MultiLabelBinarizer
    mlb_companies: MultiLabelBinarizer
    feature_columns: list

    @classmethod
    def fit(cls, df):
        df = _prepare_base_features(df)

        top_languages = df["original_language"].value_counts().nlargest(10).index.tolist()

        mlb_genres = MultiLabelBinarizer()
        mlb_genres.fit(df["genres"])

        top_countries = (
            df["production_countries"].explode().value_counts().nlargest(25).index.tolist()
        )
        mlb_countries = MultiLabelBinarizer(classes=top_countries)
        mlb_countries.fit(
            df["production_countries"].apply(lambda vals: [v for v in vals if v in top_countries])
        )

        top_companies = (
            df["production_companies"].explode().value_counts().nlargest(35).index.tolist()
        )
        mlb_companies = MultiLabelBinarizer(classes=top_companies)
        mlb_companies.fit(
            df["production_companies"].apply(lambda vals: [v for v in vals if v in top_companies])
        )

        return cls(
            top_languages=top_languages,
            mlb_genres=mlb_genres,
            mlb_countries=mlb_countries,
            mlb_companies=mlb_companies,
            feature_columns=[],
        )

    def transform(self, df):
        df = _prepare_base_features(df)

        language = df["original_language"].where(df["original_language"].isin(self.top_languages), "other")
        language = language.astype(pd.CategoricalDtype(categories=self.top_languages + ["other"]))
        language_encoded = pd.get_dummies(language, prefix="lang", dtype=int)

        genres_encoded = self.mlb_genres.transform(
            df["genres"].apply(lambda vals: [v for v in vals if v in self.mlb_genres.classes_])
        )
        genres_df = pd.DataFrame(
            genres_encoded,
            columns=[f"genre_{g}" for g in self.mlb_genres.classes_],
            index=df.index,
        )

        countries_encoded = self.mlb_countries.transform(
            df["production_countries"].apply(lambda vals: [v for v in vals if v in self.mlb_countries.classes_])
        )
        countries_df = pd.DataFrame(
            countries_encoded,
            columns=[f"country_{c}" for c in self.mlb_countries.classes_],
            index=df.index,
        )

        companies_encoded = self.mlb_companies.transform(
            df["production_companies"].apply(lambda vals: [v for v in vals if v in self.mlb_companies.classes_])
        )
        companies_df = pd.DataFrame(
            companies_encoded,
            columns=[f"company_{c}" for c in self.mlb_companies.classes_],
            index=df.index,
        )

        base_columns = ["budget", "runtime", "n_genres"]
        if "release_month" in df.columns:
            base_columns.append("release_month")

        out = pd.concat(
            [df[base_columns], language_encoded, genres_df, countries_df, companies_df],
            axis=1,
        )
        out["budget"] = np.log1p(out["budget"])

        if self.feature_columns:
            for col in self.feature_columns:
                if col not in out.columns:
                    out[col] = 0
            out = out[self.feature_columns]

        return out


def engineer_features(df):
    df = df.copy()

    # Target: se calcula acá porque depende de "revenue", que nunca debe
    # terminar siendo un feature. "revenue" es información post-estreno; si
    # quedara en X (como quedaba antes) sería leakage total, ya que el
    # target se define directamente en función de ella.
    df["profitable"] = (df["revenue"] >= df["budget"]).astype(int)
    df = df.drop(columns=["id", "adult", "revenue"])

    y = df["profitable"]
    X_raw = df.drop(columns=["profitable"])

    # Split antes de calcular cualquier estadística agregada (top-N de
    # idiomas/países/productoras): calcularlas sobre el dataset completo
    # filtraría información del test set hacia las categorías que el modelo
    # termina usando como columnas.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    encoder = FeatureEncoder.fit(X_train_raw)
    X_train = encoder.transform(X_train_raw)
    encoder.feature_columns = list(X_train.columns)
    X_test = encoder.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, encoder
