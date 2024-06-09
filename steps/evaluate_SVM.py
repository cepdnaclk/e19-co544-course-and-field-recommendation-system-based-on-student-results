import logging
from zenml import step
from sklearn.metrics import mean_squared_error

@step
def evaluate_SVM(model, X_test, y_test):
    """
    Args:
        None
    Returns:
        None
    """
    try:
        logging.info("Evaluating SVM model.")

        y_pred = model.predict(X_test)

        # Evaluate the model using mean squared error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

    except Exception as e:
        logging.error(e)
        raise e