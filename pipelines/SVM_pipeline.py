from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocess_data
from steps.train_SVM import train_SVM
from steps.evaluate_SVM import evaluate_SVM
from zenml import pipeline

@pipeline
def SVM_pipeline(data_path: str):
    """
    Ingest data, preprocess, train and evaluate MLP model.
    """
    #return df
    df = ingest_data(data_path)

    #returns X_train, X_test, y_train, y_test splitted
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_SVM(X_train, y_train)
    evaluate_SVM(model, X_test, y_test)