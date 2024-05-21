"""
This code demonstrate how to run a private LLM with private RAG in a controlled collaboration space

"""


import logging
import time
import requests
import os
import json

import duckdb

from dv_utils import default_settings, Client 

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Document
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import Document

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
    if(evt_type == "LOAD"):
        process_load_event(evt)
    else:
        generic_event_processor(evt)


def generic_event_processor(evt: dict):
    # push an audit log to reccord for an event that is not understood
    logger.info(f"Received an unhandled event {evt}")

def process_load_event(evt: dict):
    """
    Load RAG into private LLM
     """

    logger.info(f"--------------------------------------------------")
    logger.info(f"|               START CONNNECTING RAG            |")
    logger.info(f"|                                                |")
    logger.info(f"| 1. Load pretrained SentenceTransformer         |")
    # loads BAAI/bge-small-en-v1.5, embed dimension: 384, max token limit: 512
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    logger.info(f"| 1. Load data from data providers               |")
    from llama_index.core import (
        StorageContext,
        ServiceContext,
        VectorStoreIndex,
        SimpleDirectoryReader,
    )

    # DuckDB integration for storing and retrieving from knowledge base
    


    logger.info(f"|    https://github.com/./demographic.parquet |")
    logger.info(f"|    https://github.com/./patients.parquet |")
    dataProvider1URL="https://docs.google.com/document/u/0/export?format=txt&id=18SSEVxTuCmSVgM-fMcshRnNHtFVhBNMNDchxA4KqY-Q&token=AC4w5Vib0viWL7OJkKzubeJ1veTiKvq_Pg%3A1715955647070&ouid=117841664004706563551&includes_info_params=true&usp=drive_web&cros_files=false"
    dataProvider2URL="https://docs.google.com/document/u/0/export?format=txt&id=1uNB1SFT8mZ33sVGG-mFGHCSuOj3fu0ZktRDZHYyU2S8&token=AC4w5Vib0viWL7OJkKzubeJ1veTiKvq_Pg%3A1715955647070&ouid=117841664004706563551&includes_info_params=true&usp=drive_web&cros_files=false"
    start_time = time.time()
    logger.info(f"|    Start time:  {start_time} secs |")
    df = duckdb.sql("SELECT content FROM read_text(['"+dataProvider1URL+"','"+dataProvider2URL+"'])").df()
    

    documents = [Document(text=df['content'][ind]) for ind in df.index]
   
    # Set the size of the chunk to be 512 tokens
    llm = Ollama(model="llama3", base_url="http://127.0.0.1:11434",request_timeout=600.0)
    documents_service_context = ServiceContext.from_defaults(llm=llm,
    embed_model=embed_model,
    chunk_size=512)
    vector_store = DuckDBVectorStore(
        embed_dim=384,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    knowledge_base = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        service_context=documents_service_context,
    )

    # The query engine
    query_engine = knowledge_base.as_query_engine(llm=llm)

    # Run a query
    #answer = query_engine.query("Test, answer 1 word")
    #print (answer)

    # # Run a query
    # answer = query_engine.query("What are the main actions out of the board of february and April?")
    # print (answer)

    #  # Run a query
    # answer = query_engine.query("Could you give me the highlights of both board, February and April?")
    # print (answer)
   
    logger.info(f"|                                                |")
    logger.info(f"--------------------------------------------------")
   

if __name__ == "__main__":
    test_event = {
            "type": "LOAD"
    }
    process_load_event(test_event)