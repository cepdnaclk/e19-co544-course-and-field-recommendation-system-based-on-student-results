from pipelines.MLP_pipeline import MLP_pipeline

if __name__ == "__main__":
    # Run the pipeline
    data_path = "./data/Grade_CS_Students.csv"
    MLP_pipeline(data_path)