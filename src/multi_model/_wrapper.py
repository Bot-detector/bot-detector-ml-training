from pydantic import BaseModel
import pickle
from sklearn.tree import DecisionTreeClassifier
from mlflow.pyfunc.model import PythonModel, PythonModelContext
import pandas as pd
from _features import feature_engineering


class InputData(BaseModel):
    attack: int
    defence: int
    strength: int
    hitpoints: int
    ranged: int
    prayer: int
    magic: int
    cooking: int
    woodcutting: int
    fletching: int
    fishing: int
    firemaking: int
    crafting: int
    smithing: int
    mining: int
    herblore: int
    agility: int
    thieving: int
    slayer: int
    farming: int
    runecraft: int
    hunter: int
    construction: int


class OutputData(BaseModel):
    Real_Player: float
    Fletching_bot: float
    Vorkath_bot: float
    Smithing_bot: float
    Magic_bot: float
    Fishing_bot: float
    Wildy_boss_bot: float
    Wintertodt_bot: float
    Hunter_bot: float
    Blast_mine_bot: float
    Zulrah_bot: float
    Mining_bot: float
    Thieving_vyre_bot: float
    Woodcutting_bot: float
    Cooking_bot: float
    Gauntlet_bot: float
    Barrows_bot: float
    LMS_bot: float
    Crafting_bot: float
    Thieving_master_farmer_bot: float
    Doom_bot: float


class DecisionTreeWrapper(PythonModel):
    def __init__(self, params: dict, feature_fn):
        self.params = params
        self.model = DecisionTreeClassifier(**params)
        self.feature_fn = feature_fn

    def fit(self, X, y):
        """Train the decision tree model"""
        self.model.fit(X, y)
        return self

    def load_context(self, context: PythonModelContext):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def get_model(self) -> DecisionTreeClassifier:
        return self.model

    def predict(self, model_input: list[InputData], params=None) -> list[OutputData]:
        df = pd.DataFrame([d.model_dump() for d in model_input])
        df, _ = feature_engineering(df=df)
        preds = self.model.predict_proba(df)
        fields = [f for f in self.model.classes_ if isinstance(f, str)]
        return [OutputData(**{f: float(v) for f, v in zip(fields, r)}) for r in preds]
