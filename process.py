"""
This code demonstrate how to train a ML model and use it later for inference in the datavillage collaboration paradigm
The code we follow is the main example of the XGBoost ML library
https://xgboost.readthedocs.io/en/stable/get_started.html

The case is simple and very comon in machine learning 101
So you can really focus on how this example is modified to work in datavillage collaboration context

"""


import logging
import time
import requests
import os
import pandas as pd
from dv_utils import default_settings, Client 
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
import seaborn as sns
import duckdb


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


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    #audit_log("received an unhandled event", evt=evt_type)
    print ("TO DO")


def process_train_event(evt: dict):
    """
    Train an XGBoost Classifier model using the logic given in 
     """

    # load the training data from data folder
    # we could also have loaded the data from an external API or from a local file (uploaded in the data  collaboration)
    
    df = duckdb.sql("SELECT Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class FROM read_parquet('https://github.com/datavillage-me/cage-process-fraud-detection-example/raw/main/data/transactions.parquet') as transactions  JOIN read_csv(['https://github.com/datavillage-me/cage-process-fraud-detection-example/raw/main/data/label-bank1.csv','https://github.com/datavillage-me/cage-process-fraud-detection-example/raw/main/data/label-bank2.csv']) as labels ON (labels.UETR = transactions.UETR)").df()
    
    #feature_names = df.iloc[:, 1:30].columns
    #target = df.iloc[:1, 30:].columns


    #data_features = df[feature_names]
    #data_target = df[target]

    #np.random.seed(123)
    #X_train, X_test, y_train, y_test = train_test_split(data_features, data_target,train_size = 0.70, test_size = 0.30, random_state = 1)
    #xg = xgb.XGBClassifier()
    #xg.fit(X_train, y_train)

    #xg.save_model('model.json')

    #cmat, pred = RunModel(xg, X_train, y_train, X_test, y_test)
    #print(accuracy_score(y_test, pred))

    # The function "len" counts the number of classes = 1 and saves it as an object "fraud_records"
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
    xg_undersampled = xgb.XGBClassifier() 
    xg_undersampled.fit(X_undersampled_train, Y_undersampled_train)
    
    # save the model to the results location
    xg_undersampled.save_model('/resources/outputs/model.json')

    cmat, pred = RunModel(xg_undersampled, X_undersampled_train, Y_undersampled_train, X_undersampled_test, Y_undersampled_test)
    accuracy =accuracy_score(Y_undersampled_test, pred)
    
    # push the training metrics to datavillage
    client = Client()
    client.push_metrics({"accuracy":accuracy})

def RunModel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, pred)
    return matrix, pred

#if __name__ == "__main__":
 #   process_train_event(None)