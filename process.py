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
from dv_utils import default_settings, Client, audit_log 
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    elif(evt_type == "PREDICT"):
        process_predict_event(evt)
    elif(evt_type == "INFER"):
        process_infer_event(evt)     
    else:
        generic_event_processor(evt)


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    audit_log("received an unhandled event", evt=evt_type)


def process_train_event(evt: dict):
   """
   Train an XGBoost Classifier model using the logic given in 
   """

   # load the training data from scikit learn library
   # we could also have loaded the data from an external API or from a local file (uploaded in the data  collaboration)
   data = load_iris()

   # split the data in train and test samples
   X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)

   # create model instance
   bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

   # fit model using only training data
   bst.fit(X_train, y_train)

   # save the model to the results location
   bst.save_model('/resources/outputs/model.json')


   # make predictions on test dataset and verify accuracy
   preds = bst.predict(X_test)
   accuracy = accuracy_score(y_test, preds)
 
   # log the accuracy of the new model for persistence
   audit_log(f"Model was (re)trained and achieved an accuracy of {accuracy}", accuracy=accuracy)

   # push the training metrics to datavillage
   client = Client()
   client.push_metrics({"accuracy":accuracy})

def process_predict_event(evt: dict):
   """
   Make prediction using the previously train model on the test dataset 
   """

   # load the training data from scikit learn library
   # we could also have loaded the data from an external API or from a local file (uploaded in the data  collaboration)
   data = load_iris()

   # split the data in train and test samples
   X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)

   # create model instance
   bst = XGBClassifier()

   # load a previously saved model from TRAIN step from the /resources/outputs directory 
   # we could also have load a model from the /resources/inputs/ directory that would have been uploaded from a Data Provider of the collaboration
   bst.load_model('/resources/outputs/model.json')
   
   # make predictions
   preds = bst.predict(X_test)

   # save prediction and truth in a dataframe
   df = pd.DataFrame(X_test, columns=data["feature_names"])
   df["pred"] = preds
   df["pred"] = df["pred"].map(lambda x:data["target_names"][x])
   df["truth"] = y_test
   df["truth"] = df["truth"].map(lambda x:data["target_names"][x])
  
   # store predictions to excel
   df.to_excel("/resources/outputs/batch_predictions.xlsx")


def process_infer_event(evt: dict):

   # list features
   features = [
     'sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)'
   ]

   target_names = ['setosa', 'versicolor', 'virginica']


   # retrieve feature data from event
   data = evt.get("data",None)
   X = []
   for feature in features:
      if not feature in data:
          raise Exception(f"received an inference event without '{feature}' feature")
      X += [data[feature]]

   # create model instance
   bst = XGBClassifier()

   # load previously saved model
   bst.load_model('/resources/outputs/model.json')

   # make a model inference for the given features
   pred = bst.predict([X])[0]

   # get prediction name
   pred_name = target_names[pred]

   # log the inference result
   # another approach would be to push it through an API endpoint or to save it to file
   logger.info(f"Infering {pred_name} for data {data}")

