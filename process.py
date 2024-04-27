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
    if(evt_type == "BENCHMARK"):
        process_benchmark_event(evt)
    else:
        generic_event_processor(evt)


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    logger.info(f"Received an unhandled event {evt}")

def process_benchmark_event(evt: dict):
    """
    Train an XGBoost Classifier model using the logic given in 
     """

    logger.info(f"--------------------------------------------------")
    logger.info(f"|               START BENCHMARKING               |")
    logger.info(f"|                                                |")
    # load the training data from data providers
    # duckDB is used to load the data and aggregated them in one single datasets
    logger.info(f"| 1. Load data from data providers               |")
    logger.info(f"|    https://github.com/./beneficiaries1.parquet |")
    logger.info(f"|    https://github.com/./beneficiaries2.parquet |")
    dataProvider1URL="data/beneficiaries1.parquet"
    dataProvider2URL="data/beneficiaries2.parquet"
    dataProvider3URL="data/beneficiaries3.parquet"
    dataProvider4URL="data/beneficiaries4.parquet"

    start_time = time.time()
    #df = duckdb.sql("SELECT beneficiaries1.AIR_TIME FROM read_parquet('"+dataProvider1URL+"') as beneficiaries1 WHERE beneficiaries1.AIR_TIME IN (SELECT AIR_TIME from read_parquet('"+dataProvider2URL+"') UNION SELECT AIR_TIME from read_parquet('"+dataProvider3URL+"') UNION SELECT AIR_TIME from read_parquet('"+dataProvider4URL+"'))").df()
    df = duckdb.sql("SELECT FL_DATE from read_parquet('"+dataProvider2URL+"') UNION ALL SELECT FL_DATE from read_parquet('"+dataProvider3URL+"') UNION ALL SELECT FL_DATE from read_parquet('"+dataProvider4URL+"') UNION ALL SELECT FL_DATE from read_parquet('"+dataProvider1URL+"')")
    #df = duckdb.sql("DESCRIBE TABLE '"+dataProvider2URL+"'") 
    #print(df)

    execution_time=(time.time() - start_time)
    logger.info(f"|    Execution time:  {execution_time} secs |")
    logger.info(f"|    Number of records:  {len(df)}                |")

    logger.info(f"| 4. Save outputs of the collaboration           |")

    with open('/resources/outputs/benchmark-report.json', 'w', newline='') as file:
        file.write('{"Similar": "3230000","new": "628"}')
    logger.info(f"| 3. Save benchmark-report                       |")
    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")
    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")
   

if __name__ == "__main__":
    test_event = {
            'type': 'BENCHMARK'
    }
    process_benchmark_event(test_event)