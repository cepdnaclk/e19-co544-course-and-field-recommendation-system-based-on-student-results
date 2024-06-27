import logging
from zenml import step
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model._base import LinearRegression as LR
import numpy as np 
@step
def train_SVM(X_train: np.ndarray, y_train: np.ndarray) -> LR:
    """
    Args:
        X_train, y_train
    Returns:
        None
    """
    try:
        logging.info("Training SVM model.")
        model = LinearRegression()
        model.fit(X_train, y_train)

        return model

    except Exception as e:
        logging.error(e)
        raise e