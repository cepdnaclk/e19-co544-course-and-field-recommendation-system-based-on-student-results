import logging
from zenml import step
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@step
def evaluate_xgb(train_results: dict) -> dict:
    """
    Args:
        y_pred: np.array
        y_test: np.array
    Returns:
        accuracy: float
    """
    results = {}

    for column, result in train_results.items():
        y_pred = result['y_pred']
        y_test = result['y_test']

        logging.info(f"Evaluating the model for : {column}")
        
        # Calculate the evaluation criterias
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        results[f'{column}_accuracy'] = accuracy
        results[f'{column}_precision'] = precision
        results[f'{column}_recall'] = recall
        results[f'{column}_f1'] = f1

        # Display the evaluation criteria
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print('Confusion Matrix:')
        print(conf_matrix)
        print()

    #results of the evaluation without confusion matrix (if needed)
    return results