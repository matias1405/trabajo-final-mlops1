from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def create_model():
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