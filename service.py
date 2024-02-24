#service.py

import numpy as np
import bentoml
import pandas as pd

from bentoml.io import JSON
from typing import Dict, Any
from fastapi import FastAPI

from pydantic import BaseModel


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisOutput(BaseModel):
    prediction: float


iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
svc = bentoml.Service("iris_demo", runners=[iris_clf_runner])

@svc.api(
    input=JSON(pydantic_model=IrisFeatures),
    output=JSON(pydantic_model=IrisOutput),
)
def classify(input_series: IrisFeatures) -> Dict[str, Any]:
    input_df = pd.DataFrame([input_series.dict()])
    result = iris_clf_runner.predict.run(input_df).item()
    return IrisOutput(prediction=result)