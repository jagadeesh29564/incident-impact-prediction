
from functools import lru_cache
from pathlib import Path

import joblib

# Locate the saved model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "incident_model.pkl"


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


def predict_incident(description: str) -> dict:
    model = _load_model()
    prediction = model.predict([description])
    probability = model.predict_proba([description])
    classes = list(model.classes_)
    impact_class_index = classes.index(1)
    impact = "Impactful" if prediction[0] == 1 else "Non-impactful"

    return {
        "incident": description,
        "predicted_impact": impact,
        "impact_probability": float(probability[0][impact_class_index])
    }

if __name__ == "__main__":

    new_incident = (
        "Multiple customers are experiencing payment failures "
        "across production servers"
    )

    print(predict_incident("Production database unavailable"))


    print(predict_incident("Minor monitoring warning automatically resolved"))
    result = predict_incident(new_incident)


    print(result)