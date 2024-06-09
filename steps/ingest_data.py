import logging
import pandas as pd
from zenml import step


# class IngestData:
#     """
#     Data ingestion class which ingests data from the source and returns a DataFrame.
#     """

#     def __init__(self) -> None:
#         """Initialize the data ingestion class."""
#         pass

#     def get_data(self) -> pd.DataFrame:
#         df = pd.read_csv("./data/Grade_CS_Students.csv")
#         return df


@step
def ingest_data(data_path):

    try:
        logging.info("Ingesting data")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logging.error(e)
        raise e
