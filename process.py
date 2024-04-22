"""
This code demonstrate how to train a XGBoost Logistic Regression model for credit card fraud detection
The code use datasets from 3 parties
- 2 banks providing the labels (class) for each transactions being fraudulent or not
- A financial data intermediary or payment processor providing the transactions data on which Dimensionality Reduction Techniques for Data Privacy has been applied.

"""


import logging
import time
import requests
import os
import json

import duckdb

import pandas as pd
from dv_utils import default_settings, Client 
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# let the log go to stdout, as it will be captured by the cage operator
logging.basicConfig(
    level=default_settings.log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# define an event processing function
def event_processor(evt: dict):
    """
    Process an incoming event
    Exception raised by this function are handled by the default event listener and reported in the logs.
    """
    
    logger.info(f"Processing event {evt}")

    # dispatch events according to their type
    evt_type =evt.get("type", "")
    if(evt_type == "TRAIN"):
        process_train_event(evt)
    elif(evt_type == "INFER"):
        process_infer_event(evt)
    else:
        generic_event_processor(evt)


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    logger.info(f"Received an unhandled event {evt}")

def process_train_event(evt: dict):
    """
    Train an XGBoost Classifier model using the logic given in 
     """

    # load the training data from data providers
    # duckDB is used to load the data and aggregated them in one single datasets
    logger.info(f"Load data from data providers")
    df = duckdb.sql("SELECT Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class FROM read_parquet('https://github.com/datavillage-me/cage-process-fraud-detection-example/raw/main/data/transactions.parquet') as transactions  JOIN read_csv(['https://github.com/datavillage-me/cage-process-fraud-detection-example/raw/main/data/label-bank1.csv','https://github.com/datavillage-me/cage-process-fraud-detection-example/raw/main/data/label-bank2.csv']) as labels ON (labels.UETR = transactions.UETR)").df()
    
    # Defines number of fraud recors
    fraud_records = len(df[df.Class == 1]) 

    # Defines the index for fraud and non-fraud in the lines:
    fraud_indices = df[df.Class == 1].index
    normal_indices = df[df.Class == 0].index

    # Randomly collect equal samples of each type:
    under_sample_indices = np.random.choice(normal_indices, fraud_records, False)
    df_undersampled = df.iloc[np.concatenate([fraud_indices, under_sample_indices]),:]
    X_undersampled = df_undersampled.iloc[:,1:30]
    Y_undersampled = df_undersampled.Class
    X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = train_test_split(X_undersampled, Y_undersampled, test_size = 0.30)
    logger.info(f"Create XGBClassifier model")
    xg_undersampled = xgb.XGBClassifier() 
    xg_undersampled.fit(X_undersampled_train, Y_undersampled_train)

    logger.info(f"Run model")
    cmat, pred = RunModel(xg_undersampled, X_undersampled_train, Y_undersampled_train, X_undersampled_test, Y_undersampled_test)
    accuracy =accuracy_score(Y_undersampled_test, pred)
    classificationReportJson=classification_report(Y_undersampled_test, pred,output_dict=True)

    logger.info(f"Save model as output of the collaboration")
    # save the model to the results location
    xg_undersampled.save_model('/resources/outputs/model.json')
    
    logger.info(f"Save model classification report as output of the collaboration")
    with open('/resources/outputs/classification-report.json', 'w', newline='') as file:
       file.write(json.dumps(classificationReportJson))
    
    logger.info(f"Log accuracy")
    # push the training metrics to datavillage
    client = Client()
    client.push_metrics({"accuracy":accuracy})

def process_infer_event(evt: dict):
    """
    Infer prediction using the previously train model on the data from the event
    """

    features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

    # retrieve feature data from event
    data = evt.get("data",None)
    X = []
    for feature in features:
        if not feature in data:
            raise Exception(f"received an inference event without '{feature}' feature")
        X += [data[feature]]
    # create model instance
    bst = xgb.XGBClassifier()

     # load previously saved model
    bst.load_model('/resources/outputs/model.json')

    # make a model inference for the given features
    pred = bst.predict([X])[0]

    print (pred)



def RunModel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, pred)
    return matrix, pred

if __name__ == "__main__":
    test_event = {
            'type': 'INFER',
            'data': 
        {
    "V1": 1.23742903021294,
    "V2": -0.93765432109876,
    "V3": 0.54218765432109,
    "V4": -1.23568765432109,
    "V5": 0.76543210987654,
    "V6": -0.98765432109876,
    "V7": 1.12345678901234,
    "V8": -0.78901234567890,
    "V9": 1.54321098765432,
    "V10": -1.23456789012345,
    "V11": 0.87654321098765,
    "V12": -0.65432109876543,
    "V13": 1.09876543210987,
    "V14": -0.43210987654321,
    "V15": 0.21098765432109,
    "V16": -0.34567890123456,
    "V17": 0.12345678901234,
    "V18": -0.56789012345678,
    "V19": 0.78901234567890,
    "V20": -0.98765432109876,
    "V21": 0.65432109876543,
    "V22": -0.32109876543210,
    "V23": 0.09876543210987,
    "V24": -0.45678901234567,
    "V25": 0.23456789012345,
    "V26": -0.54321098765432,
    "V27": 0.87654321098765,
    "V28": -0.12345678901234,
    "Amount": 149.62
         }  
    
    }
    process_infer_event(test_event)