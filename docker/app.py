
import pickle
import numpy as np
import pandas as pd
import os
import json

# Import Flask for creating API
from flask import Flask, request

with open('./outputs/diabetes-rf.pkl', 'rb') as model_pkl:
    knn = pickle.load(model_pkl)

columns= ['race', 'gender', 'discharge_disposition_id', 'admission_source_id',
       'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum',
       'A1Cresult', 'metformin', 'glimepiride', 'glipizide', 'glyburide',
       'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed',
       'age', 'time_in_hospital', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'requests_1', 'requests_10']

# Initialise a Flask app
port = int(os.environ.get("PORT", 3000))
app = Flask(__name__)

# Create an API endpoint
@app.route('/predict')
def predict_readmission():

    # http://localhost:3000/predict?race='Caucasian'&gender='Male'&discharge_disposition_id=11&admission_source_id=1&medical_specialty='Other'&diag_1=250.7&diag_2=410&diag_3=785&max_glu_serum=None&A1Cresult=None&metformin='No'&glimepiride='No'&glipizide='No'&glyburide='No'&pioglitazone='No'&rosiglitazone='No'&insulin='Steady'&change='No'&diabetesMed='Yes'&age=55&time_in_hospital=2&num_medications=24&number_outpatient=0&number_emergency=0&number_inpatient=0&requests_1=0&requests_10=0
    # Read all necessary request parameters

    race= request.args.get('race')
    gender= request.args.get('gender')
    discharge_disposition_id= request.args.get('discharge_disposition_id')
    admission_source_id= request.args.get('admission_source_id')
    medical_specialty= request.args.get('medical_specialty')
    diag_1= request.args.get('diag_1')
    diag_2= request.args.get('diag_2')
    diag_3= request.args.get('diag_3')
    max_glu_serum= request.args.get('max_glu_serum')
    A1Cresult= request.args.get('A1Cresult')
    metformin= request.args.get('metformin')
    glimepiride= request.args.get('glimepiride')
    glipizide= request.args.get('glipizide')
    glyburide= request.args.get('glyburide')
    pioglitazone= request.args.get('pioglitazone')
    rosiglitazone= request.args.get('rosiglitazone')
    insulin= request.args.get('insulin')
    change= request.args.get('change')
    diabetesMed= request.args.get('diabetesMed')
    age= request.args.get('age')
    time_in_hospital= request.args.get('time_in_hospital')
    num_medications= request.args.get('num_medications')
    number_outpatient= request.args.get('number_outpatient')
    number_emergency= request.args.get('number_emergency')
    number_inpatient= request.args.get('number_inpatient')
    requests_1= request.args.get('requests_1')
    requests_10= request.args.get('requests_10')

    data = np.array([[race,
                            gender,
                            discharge_disposition_id,
                            admission_source_id,
                            medical_specialty,
                            diag_1,
                            diag_2,
                            diag_3,
                            max_glu_serum,
                            A1Cresult,
                            metformin,
                            glimepiride,
                            glipizide,
                            glyburide,
                            pioglitazone,
                            rosiglitazone,
                            insulin,
                            change,
                            diabetesMed,
                            age,
                            time_in_hospital,
                            num_medications,
                            number_outpatient,
                            number_emergency,
                            number_inpatient,
                            requests_1,
                            requests_10]])
    new_record = pd.DataFrame(data, index=None, columns=columns)
    predict_result = knn.predict(new_record)
    readmission = ['No', 'Yes']
    # return the result back
    result = json.dumps({'Will the patien be readmited?' : readmission[int(predict_result)]})
    return result

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=port)