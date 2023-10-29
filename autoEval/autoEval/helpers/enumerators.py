from enum import Enum


class ModellingTaskType(Enum):
    Regression = "Regression"
    Classification = "Classification"
    Forecasting = "Forecasting"
