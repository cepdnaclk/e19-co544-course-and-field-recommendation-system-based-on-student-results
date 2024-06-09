from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocess_data
from steps.train_MLP import train_MLP
from steps.evaluate_MLP import evaluate_MLP
from zenml import pipeline

@pipeline
def MLP_pipeline(data_path: str):
    """
    Ingest data, preprocess, train and evaluate MLP model.
    """
    df = ingest_data(data_path)
    preprocess_data(df)
    train_MLP(df)
    evaluate_MLP(df)