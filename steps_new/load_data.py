import pandas as pd
from zenml import step
import logging

@step(enable_cache=False)
def load_data(data_path: str) -> pd.DataFrame:
    """
    Args:
        data_path: str
    Returns:
        df: pd.DataFrame
    """
    try:
        logging.info("Loading data.")
        df = pd.read_csv(data_path, na_values=['NA'])
        # df = pd.read_excel(data_path, na_values=['NA'])
        return df
    
    except Exception as e:
        logging.error(e)
        raise e