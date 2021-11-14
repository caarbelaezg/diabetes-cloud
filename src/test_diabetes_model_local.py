import pickle
import pandas as pd
import numpy as np
import json

#load the model

with open('../outputs/diabetes-rf.pkl', 'rb') as model_pkl:
    rf = pickle.load(model_pkl)

if __name__ == "__main__":
    columns= ['race', 'gender', 'discharge_disposition_id', 'admission_source_id',
       'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum',
       'A1Cresult', 'metformin', 'glimepiride', 'glipizide', 'glyburide',
       'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed',
       'age', 'time_in_hospital', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'requests_1', 'requests_10']

    data = np.array([['Caucasian','Male',11,1,'Other',250.7,410,785,None,None,'No','No','No','No','No','No','Steady','No','Yes',55,2,24,0,0,0,0,0]])
    new_record = pd.DataFrame(data, index=None, columns=columns)
    predicted_result = rf.predict(new_record)
    readmission = ['No', 'Yes']

    result = json.dumps({'Will the patien be readmited?' : readmission[int(predicted_result)]})
    print(result)