from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold


def create_random_forest():
    """Crea el modelo V1 productivo basado en Random Forest."""
    return Pipeline([
        (
            "variance_filter",
            VarianceThreshold(threshold=0.005),
        ),
        (
            "clf",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def create_svm():
    """Crea el modelo V2-RC basado en SVM."""
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "clf",
            CalibratedClassifierCV(
                SVC(
                    class_weight="balanced",
                    random_state=42,
                ),
                ensemble=False,
            ),
        ),
    ])