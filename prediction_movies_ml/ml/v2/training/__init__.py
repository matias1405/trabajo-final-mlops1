"""Entrenamiento y evaluación del modelo de rentabilidad de películas, y
registro del resultado en el Model Registry de MLflow.
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, StandardScaler
import numpy as np
import pandas as pd

class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_top_languages=10,
        n_top_countries=25,
        n_top_companies=35
    ):
        self.n_top_languages = n_top_languages
        self.n_top_countries = n_top_countries
        self.n_top_companies = n_top_companies


    def _to_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if pd.isna(value):
            return []
        return [v.strip() for v in str(value).split(",") if v.strip()]


    def _convert_to_lists(self, df):
        df = df.copy()
        for col in ["genres","production_countries","production_companies"]:
            df[col] = df[col].apply(self._to_list)
        return df
    

    def fit(self, X, y=None):
        X = self._convert_to_lists(X)
        # original_language
        self.top_languages_ = (X["original_language"].value_counts().nlargest(self.n_top_languages).index.tolist())
        # genres
        self.mlb_genres_ = MultiLabelBinarizer()
        self.mlb_genres_.fit(X["genres"])
        # production_countries
        self.top_countries_ = (X["production_countries"].explode().value_counts().nlargest(self.n_top_countries).index.tolist())
        self.mlb_countries_ = MultiLabelBinarizer(classes=self.top_countries_)
        self.mlb_countries_.fit(X["production_countries"])
        # production_companies
        self.top_companies_ = (X["production_companies"].explode().value_counts().nlargest(self.n_top_companies).index.tolist())
        self.mlb_companies_ = MultiLabelBinarizer(classes=self.top_companies_)
        self.mlb_companies_.fit(X["production_companies"])
        return self

    def transform(self, X):
        X = self._convert_to_lists(X)
        # cantidad de géneros
        X["n_genres"] = X["genres"].apply(len)
        # original_language
        X["original_language"] = X["original_language"].where(
            X["original_language"].isin(self.top_languages_),"other")
        language_encoded = pd.get_dummies(X["original_language"],prefix="lang",dtype=int)

        # Garantizar exactamente las mismas
        # columnas que durante training
        expected_languages = [f"lang_{lang}" for lang in self.top_languages_] + ["lang_other"]
        language_encoded = language_encoded.reindex(columns=expected_languages,fill_value=0)
        X = pd.concat([X.drop(columns=["original_language"]),language_encoded],axis=1)
        # genres - multi-hot
        genres_encoded = self.mlb_genres_.transform(X["genres"])
        genres_df = pd.DataFrame(genres_encoded,
            columns=[f"genre_{g}" for g in self.mlb_genres_.classes_],
            index=X.index
        )
        X = pd.concat([X.drop(columns=["genres"]),genres_df],axis=1)
        # production_countries - multi-hot
        countries = X["production_countries"].apply(
            lambda values: [c for c in values if c in self.top_countries_]
        )
        countries_encoded = self.mlb_countries_.transform(countries)
        countries_df = pd.DataFrame(countries_encoded,
            columns=[f"country_{c}" for c in self.mlb_countries_.classes_],
            index=X.index
        )
        X = pd.concat([X.drop(columns=["production_countries"]), countries_df],axis=1)
        # production_companies - multi-hot
        companies = X["production_companies"].apply(
            lambda values: [c for c in values if c in self.top_companies_]
        )
        companies_encoded = self.mlb_companies_.transform(companies)
        companies_df = pd.DataFrame(companies_encoded,
            columns=[f"company_{c}" for c in self.mlb_companies_.classes_],
            index=X.index
        )
        X = pd.concat([X.drop(columns=["production_companies"]),companies_df],axis=1)
        # budget
        X["budget"] = np.log1p(X["budget"])
        # release_month
        X["release_month"] = (pd.to_datetime(X["release_date"]).dt.month)
        X = X.drop(columns=["release_date"])
        # dejar solamente features numéricas
        X = X.select_dtypes(include=["number", "bool"])
        return X

def train(x_train, y_train): 
    model = Pipeline([
        ("features", FeatureEngineering()),
        ('scaler', StandardScaler()),
        ("clf",CalibratedClassifierCV(SVC(class_weight="balanced",random_state=14),ensemble=False))
    ])
    model.fit(x_train, y_train)
    return model


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }


def register_model(model, run_metrics, model_name="PredictionMovies", alias="staging"):
    with mlflow.start_run():
        mlflow.log_metrics(run_metrics)
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        version = model_info.registered_model_version
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(model_name, alias, version)
        version_tag = f"v{version}-RC"
        client.set_model_version_tag(model_name,version,"release",version_tag)
    return version
