import logging
from zenml import step
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

@step
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: pd.DataFrame
    Returns:
        df: pd.DataFrame
    """
    try:
        logging.info("Preprocessing data.")
        # Replace 'NA' with NaN and convert columns to numeric where applicable
        df.replace('NA', pd.NA, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Define features (marks from 1st to 4th year) and target variable (marks of 5th-year courses)
        features = ['CS101', 'CS102', 'MA101', 'MA112', 'MA121', 'GS101', 'GS102', 'PH101',
                    'PH102', 'PH103', 'PH104', 'ST111', 'ST112', 'ST121', 'ST122', 'ST132',
                    'CS201', 'CS202', 'CS203', 'CS204', 'MA201', 'MA202', 'MA211', 'MA212',
                    'MA272', 'GS202', 'GS203', 'OR201', 'OR202', 'ST203', 'ST204', 'ST211',
                    'CS301', 'CS302', 'CS303', 'CS304', 'CS305', 'CS306', 'CS307', 'CS308',
                    'CS309', 'CS311', 'CS312', 'CS314', 'MA307', 'MA308', 'MA381', 'MA382',
                    'MA392', 'CS497', 'MA419', 'SW499']
        target_courses = ['CS501', 'CS502', 'CS503', 'CS504', 'CS505', 'CS506', 'CS507', 'CS508', 'CS509', 'CS510', 'CS512']

        # Split the dataset into features and target variable
        X = df[features]
        y = df[target_courses]

        # Impute missing values using mean imputation for both features and target variable
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        y_imputed = imputer.fit_transform(y)

        # Split the dataset into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(e)
        raise e
    




