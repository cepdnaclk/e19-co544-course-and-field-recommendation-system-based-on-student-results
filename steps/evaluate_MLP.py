import logging
from zenml import step

@step
def evaluate_MLP():
    """
    Args:
        None
    Returns:
        None
    """
    try:
        logging.info("Evaluating MLP model.")
    except Exception as e:
        logging.error(e)
        raise e