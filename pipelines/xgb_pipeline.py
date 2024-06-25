from steps_new.load_data import load_data
from steps_new.preprocess_data import preprocess_data
from steps_new.train_model import train_xgboost_for_each_target
from steps_new.evaluate_model import evaluate_xgb
from zenml import pipeline

@pipeline
def XGB_pipeline(data_path: str):
    """
    Ingest data, preprocess, train and evaluate MLP model.
    """
    df = load_data(data_path)
    features, targets = preprocess_data(df)
    train_results_dict = train_xgboost_for_each_target(features, targets)

    #implement correct DS from train
    evaluate_xgb(train_results_dict)