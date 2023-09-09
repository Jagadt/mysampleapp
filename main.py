#reference
#https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Langchain%20Chatbot/utils.py
#https://medium.com/google-cloud/deploy-your-custom-knowledge-base-assistant-powered-by-vertex-ai-and-pinecone-c4e8f4868b99

import streamlit as st
from streamlit_chat import message
from utils import *
import os
from langchain.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from google.cloud import aiplatform

from google.auth import credentials
from google.oauth2 import service_account

#google auth
key_path = 'C:/Users/Jagadeesh/Documents/mygithub/gcp/gcpkey.json'
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

PROJECT_NAME = os.environ.get("PROJECT_NAME")
credentials = service_account.Credentials.from_service_account_file("C:/Users/Jagadeesh/Documents/mygithub/gcp/gcpkey.json")
aiplatform.init(project="PROJECT_NAME", credentials=credentials)

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


#Page Config
st.set_page_config(
     layout="wide",
     page_title="JS Lab",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"
)

#Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    "A place for me to experiment different LLM use cases, models, application frameworks and etc."
)

#Main Page and Chatbot components
st.title("Internal Knowledge Base Chatbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I help you today?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatVertexAI(model_name="chat-bison")
if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'Sorry! I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_history = get_conversation_history()
            refined_query = query_refiner(conversation_history, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')