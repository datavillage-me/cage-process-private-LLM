# this code can be tested in a local environment
# for this, start a local redis server and set the appropriate variables in .env.test

# Read env variables from a local .env file, to fake the variables normally provided by the cage container
import dotenv
dotenv.load_dotenv('.env.test')
import os
import unittest
import process
import json
import logging
from dv_utils import RedisQueue

DEBUG = os.environ.get('DEBUG', '').lower() == 'true'
logging.basicConfig(filename='events.log', level=logging.DEBUG if DEBUG else logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class Test(unittest.TestCase):

    def test_process(self):
        """
        Try the process on a single user configured in the test .env file, without going through the redis queue
        """
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

    def test_process_queue_once(self):
        """
        Try the process by sending an event to the queue and consume exactly one event.
        """

        test_event = {
            'userIds': [os.environ["TEST_USER"]],
            'trigger': 'full',
            # 'jobId': None,
        }

        r = RedisQueue()

        # fake the publishing of an event
        r.redis.publish("dv", json.dumps(test_event))

        # wait for exactly one event and process it
        event = r.listenOnce()
        process.event_processor(event)

    def test_process_queue_loop(self):
        """
        Try the process by sending an event to the queue and let the queue loop in wait.
        """

        test_event = {
            'userIds': [os.environ["TEST_USER"]],
            'trigger': 'full',
            # 'jobId': None,
        }

        r = RedisQueue()

        # fake the publishing of an event
        r.redis.publish("dv", json.dumps(test_event))

        # let the queue wait for events and dispatch them to the process function
        r.listen(process.event_processor, 2)
