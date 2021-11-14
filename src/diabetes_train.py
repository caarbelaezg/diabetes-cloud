import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import pickle

if __name__ == "__main__":    

    #Datasets: Prep
    PATH = os.path.join('..', 'data', 'cleanData.csv')
    df = pd.read_csv(PATH, index_col=[0])

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
    
    x_test.to_csv('../data/x_test.csv')
    y_test.to_csv('../data/y_test.csv')

    #ML Model: Random forest classifier

    rf1 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
    rf1.fit(x_train, y_train)

    #Predictions
    predictions=rf1.predict(x_test)

    #Accuracy
    print(accuracy_score(y_test, predictions))

    #Registro
    OUTPUT_PATH = os.path.join('..', 'outputs', 'diabetes-rf.pkl')
    with open(OUTPUT_PATH, 'wb') as model_pkl:
        pickle.dump(rf1, model_pkl)