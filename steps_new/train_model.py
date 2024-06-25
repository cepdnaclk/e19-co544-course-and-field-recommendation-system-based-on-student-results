import logging
from sklearn.preprocessing import LabelEncoder
from zenml import step
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

@step
def train_xgboost_for_each_target(features_encoded, targets_encoded):
    """
    Args:
        features_encoded: pd.DataFrame
        targets_encoded: pd.DataFrame
    Returns:
        results: dict
    """
    results = {}

    for column in targets_encoded.columns:

        logging.info(f"Training XGBoost model for target column: {column}")

        smote = SMOTE(k_neighbors=3)
        features_resampled, target_resampled = smote.fit_resample(
            features_encoded, targets_encoded[column])

        X_train, X_test, y_train, y_test = train_test_split(
            features_resampled, target_resampled, test_size=0.2, random_state=42)
        
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        xgb = XGBClassifier()

        # Train the classifier
        xgb.fit(X_train, y_train)

        # Predict the target values
        y_pred = xgb.predict(X_test)

        results[column] = {
            'y_test': y_test,
            'y_pred': y_pred
        }

    return results