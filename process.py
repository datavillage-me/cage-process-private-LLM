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

from dv_utils import default_settings, Client 

import pandas as pd

logger = logging.getLogger(__name__)

input_dir = "/resources/data"
output_dir = "/resources/outputs"

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
    if(evt_type == "QUERY"):
        process_query_event(evt)
    else:
        generic_event_processor(evt)


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    logger.info(f"Received an unhandled event {evt}")

def process_query_event(evt: dict):
    """
    Train an XGBoost Classifier model using the logic given in 
     """

    logger.info(f"--------------------------------------------------")
    logger.info(f"|               START BENCHMARKING               |")
    logger.info(f"|                                                |")
    # load the training data from data providers
    # duckDB is used to load the data and aggregated them in one single datasets
    logger.info(f"| 1. Load data from data providers               |")
    logger.info(f"|    https://github.com/./demographic.parquet |")
    logger.info(f"|    https://github.com/./patients.parquet |")
    dataProvider1URL="https://github.com/datavillage-me/cage-process-clinical-trial-patient-cohort-selection/raw/main/data/demographic.parquet"
    #dataProvider1URL="data/demographic.parquet"
    dataProvider2URL="https://github.com/datavillage-me/cage-process-clinical-trial-patient-cohort-selection/raw/main/data/patients.parquet"
    #dataProvider2URL="data/patients.parquet"
    start_time = time.time()
    logger.info(f"|    Start time:  {start_time} secs |")
    
    whereClause=evt.get("parameters", "")
    if whereClause!='':
        baseQuery="SELECT COUNT(*) as total from '"+dataProvider1URL+"' as demographic,'"+dataProvider2URL+"' as patients WHERE demographic.national_id=patients.national_id AND "+whereClause
    else:
        baseQuery="SELECT COUNT(*) as total from '"+dataProvider1URL+"' as demographic,'"+dataProvider2URL+"' as patients WHERE demographic.national_id=patients.national_id"
    
    #total candidates
    df = duckdb.sql(baseQuery).df()
    totalCandidates=df['total'][0]
    print(totalCandidates)
    
    #gender
    #male
    df = duckdb.sql(baseQuery+ " AND demographic.gender='male'").df()
    totalGenderMale=df['total'][0]
    #female
    df = duckdb.sql(baseQuery+ " AND demographic.gender='female'").df()
    totalGenderFemale=df['total'][0]

    #education_level
    #high_school
    df = duckdb.sql(baseQuery+ " AND demographic.education_level='high_school'").df()
    totalEducationLevelHighSchool=df['total'][0]
    #college
    df = duckdb.sql(baseQuery+ " AND demographic.education_level='college'").df()
    totalEducationLevelCollege=df['total'][0]
    #university
    df = duckdb.sql(baseQuery+ " AND demographic.education_level='university'").df()
    totalEducationLevelUniversity=df['total'][0]


    #employment_status
    #unemployed
    df = duckdb.sql(baseQuery+ " AND demographic.employment_status='unemployed'").df()
    totalEmploymentStatusUnemployed=df['total'][0]
    #employed
    df = duckdb.sql(baseQuery+ " AND demographic.employment_status='employed'").df()
    totalEmploymentStatusEmployed=df['total'][0]
    #student
    df = duckdb.sql(baseQuery+ " AND demographic.employment_status='student'").df()
    totalEmploymentStatusStudent=df['total'][0]
    #retired
    df = duckdb.sql(baseQuery+ " AND demographic.employment_status='retired'").df()
    totalEmploymentStatusRetired=df['total'][0]

    execution_time=(time.time() - start_time)
    logger.info(f"|    Execution time:  {execution_time} secs |")

    logger.info(f"| 4. Save outputs of the collaboration           |")
    # The output file model is stored in the data folder
    
    output= ''' {
    "candidates": '''+str(totalCandidates)+''',
        "gender": {
        "male":'''+str(totalGenderMale)+''',
        "female":'''+str(totalGenderFemale)+'''
        },
        "education_level": {
        "high_school":'''+str(totalEducationLevelHighSchool)+''',
        "college":'''+str(totalEducationLevelCollege)+''',
        "university":'''+str(totalEducationLevelUniversity)+'''
        },
        "employment_status":{
        "unemployed":'''+str(totalEmploymentStatusUnemployed)+''',
        "employed":'''+str(totalEmploymentStatusEmployed)+''',
        "student":'''+str(totalEmploymentStatusStudent)+''',
        "retired":'''+str(totalEmploymentStatusRetired)+'''
        }
    } '''

    #with open('data/my.json', 'w', newline='') as file:
        #file.write(output)

    with open('/resources/outputs/candidates-report.json', 'w', newline='') as file:
        file.write(output)
    logger.info(f"| 3. Save candidate-report                       |")
    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")
    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")
   

if __name__ == "__main__":
    test_event = {
            "type": "QUERY",
            "parameters": ""
    }
    process_query_event(test_event)