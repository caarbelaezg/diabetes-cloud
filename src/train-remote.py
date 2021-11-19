import os
import argparse
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import pickle
from azureml.core import Run

run = Run.get_context()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='.',
        help='Path to the training data'
    )

    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    PATH = os.path.join(args.data_path, os.listdir(args.data_path)[0])
    print("================")
    print(f'DataPath: {PATH}')
    df = pd.read_csv(PATH, index_col=[0])

    # prepare diabetes data
    cat_features = ['race', 'gender', 'discharge_disposition_id', 'admission_source_id',
                'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum',
                'A1Cresult', 'metformin', 'glimepiride', 'glipizide', 'glyburide',
                'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed']

    num_features = ['age', 'time_in_hospital', 'num_medications', 'number_outpatient',
                'number_emergency', 'number_inpatient', 'requests_1', 'requests_10']

    y = ['readmitted']

    x = cat_features + num_features

    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())])

    categorical_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)])


    #Train and test data
    x_train, x_test, y_train, y_test = train_test_split(df[x], df[y], test_size=0.30, random_state=27)
    

    #ML Model: Random forest classifier

    rf1 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
    rf1.fit(x_train, y_train)
    predictions=rf1.predict(x_test)

    print(accuracy_score(y_test, predictions))
    run.log('Accuracy model', accuracy_score(y_test, predictions))

    Pkl_Filename = "outputs/diabetes_model.pkl"  

    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(rf1, file)
    print('Finished Training')