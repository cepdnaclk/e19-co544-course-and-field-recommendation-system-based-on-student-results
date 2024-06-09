import logging
from zenml import step
from sklearn.linear_model import LinearRegression

@step
def train_SVM(X_train, y_train):
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