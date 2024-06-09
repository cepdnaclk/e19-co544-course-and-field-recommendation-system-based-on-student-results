import logging
from zenml import step

@step
def train_MLP():
    """
    Args:
        None
    Returns:
        None
    """
    try:
        logging.info("Training MLP model.")
    except Exception as e:
        logging.error(e)
        raise e