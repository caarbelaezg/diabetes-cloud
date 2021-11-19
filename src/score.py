import json
import numpy as np
import pandas as pd
import os
import joblib

def init():
    global model

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'diabetes_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    columns= ['race', 'gender', 'discharge_disposition_id', 'admission_source_id',
       'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum',
       'A1Cresult', 'metformin', 'glimepiride', 'glipizide', 'glyburide',
       'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed',
       'age', 'time_in_hospital', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'requests_1', 'requests_10']
    data = np.array(json.loads(raw_data)['data'])
    new_record = pd.DataFrame(data, index=None, columns=columns)    
    y_hat = model.predict(new_record)
    readmission = ['No', 'Yes']
    return json.dumps({'Will the patien be readmited?' : readmission[int(y_hat)]})