import pickle

import pandas as pd
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from lightgbm import LGBMClassifier

from _features import feature_engineering
from _structs import InputData, OutputData


class LGBMWrapper(PythonModel):
    def __init__(self, params: dict):
        self.params = params
        self.model = LGBMClassifier(**params)

    @classmethod
    def from_lgbm(cls, model: LGBMClassifier):
        """Create LGBMWrapper from an existing LGBMClassifier."""
        params = model.get_params()
        wrapper = cls(params)
        wrapper.model = model
        return wrapper

    def fit(self, X, y):
        """Train the decision tree model"""
        self.model.fit(X, y)
        return self

    def load_context(self, context: PythonModelContext):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def get_model(self) -> LGBMClassifier:
        return self.model

    def get_input_json_schema(self):
        return InputData.model_json_schema()

    def get_output_json_schema(self):
        return OutputData.model_json_schema()

    def predict(self, model_input: list[InputData], params=None) -> list[OutputData]:
        df = pd.DataFrame([d.model_dump() for d in model_input])
        df, _ = feature_engineering(df=df)
        preds = self.model.predict_proba(df)
        fields = [f for f in self.model.classes_ if isinstance(f, str)]
        return [OutputData(**{f: float(v) for f, v in zip(fields, r)}) for r in preds]
