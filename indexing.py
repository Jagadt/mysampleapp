import nltk
import functions_framework
import os
import pinecone
from langchain.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv


load_dotenv('C:/Users/Jagadeesh/Documents/mygithub/gcp/.env')

#PROJECT_NAME ="savvy-equator-396018"
#BUCKET_NAME = "myenterprisesearchdata"
#API_KEY = "2c313983-b885-44cc-95bf-ab3aec41763b"
#REGION_INFO = "gcp-starter"
#INDEX_NAME = "mypineconeind"

PROJECT_NAME = os.environ.get("PROJECT_NAME")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
API_KEY = os.environ.get("API_KEY")
REGION_INFO = os.environ.get("REGION_INFO")
INDEX_NAME = os.environ.get("INDEX_NAME")


#function to load documents from GCS bucket
def load_docs(projectID, bucketName):
    loader = GCSDirectoryLoader(project_name=projectID, bucket=bucketName)
    documents = loader.load()
    return documents

#function to split documents into chunk size
def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

#gcf entry point
@functions_framework.cloud_event
def new_object(cloud_event):
  data=cloud_event.data
  print(cloud_event["id"])
  documents = load_docs("PROJECT_NAME", "BUCKET_NAME")
  docs = split_docs(documents)
  embeddings = VertexAIEmbeddings(model_name="textembedding-gecko")
  pinecone.init(
    api_key="API_KEY",  # find at app.pinecone.io
    environment="REGION_INFO"  # next to api key in console
    )
  index_name = "INDEX_NAME"
  index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
  