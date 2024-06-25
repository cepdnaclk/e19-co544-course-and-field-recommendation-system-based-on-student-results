from pipelines.xgb_pipeline import XGB_pipeline
import logging

if __name__ == "__main__":
    
    # Run the pipeline
    data_path = "./data/Grade_CS_Students.csv"
    logging.info("Starting XGB pipeline")
    XGB_pipeline(data_path)