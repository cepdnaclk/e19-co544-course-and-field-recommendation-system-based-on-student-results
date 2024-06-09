from pipelines.SVM_pipeline import SVM_pipeline
import pandas as pd
import logging

if __name__ == "__main__":
    # Run the pipeline
    data_path = "./data/Grade_CS_Students.csv"
    logging.info("Starting SVM pipeline")
    SVM_pipeline(data_path)