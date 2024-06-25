import logging
from zenml import step
import pandas as pd


def encode_grade(marks):

    '''Encode the marks into grades'''	

# Convert scores into grades

    if marks > 85:
        return 0  #'A+'
    elif 80 <= marks <= 85:
        return 1  #'A'
    elif 75 <= marks < 80:
        return 2  #'A-'
    elif 70 <= marks < 75:
        return 3  #'B+'
    elif 65 <= marks < 70:
        return 4  #'B'
    elif 60 <= marks < 65:
        return 5  #'B-'
    elif 55 <= marks < 60:
        return 6  #'C+'
    elif 50 <= marks < 55:
        return 7  #'C'
    elif 45 <= marks < 50:
        return 8  #'C-'
    elif 40 <= marks < 45:
        return 9 #'D+'
    elif 35 <= marks < 40:
        return 10 #'D'
    else:
        return 11 #'E'

def merge_small_classes(target_column, threshold=4):
    '''Function to merge small classes into the next larger class'''

    value_counts = target_column.value_counts()
    small_classes = value_counts[value_counts < threshold].index
    for small_class in small_classes:
        # Find the next larger class
        larger_classes = value_counts[value_counts >= threshold].index
        if len(larger_classes) > 0:
            next_larger_class = larger_classes[0]
            target_column[target_column == small_class] = next_larger_class
        else:
            # If no larger class exists, keep the class as is
            continue
    return target_column

@step	
def preprocess_data(df):

    '''
    Preprocess the data
    input: df: pd.DataFrame
    output: 
        features_encoded: pd.DataFrame, 
        targets_encoded: pd.DataFrame
    '''

    try:
        # drop unnecessary columns
        df = df.drop(['Year of enrolment', 'ID'], axis=1)

        # divide features and targets
        target_subjects = ['CS501','CS502','CS503','CS504','CS505','CS506','CS507',
                        'CS508','CS509','CS510','CS512','CS597','CS598','MM507'] 
        features = df.drop(target_subjects, axis=1)
        targets = df[target_subjects]

        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')  # Convert to numeric, set non-numeric to NaN
            features.fillna({col: features[col].median()}, inplace=True)  # Fill NaN with median of the column

        # Apply the grade encoding function to each cell in the dataframe
        features_encoded = features.map(encode_grade)
        targets_encoded = targets.map(encode_grade)

        for column in targets_encoded.columns:
            # Merge small classes in the target column
            targets_encoded[column] = merge_small_classes(targets_encoded[column])

        return features_encoded, targets_encoded
    
    except Exception as e:
        logging.error(e)
        raise e
