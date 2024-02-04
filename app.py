from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import streamlit as st
import os

# This loads all environment variables from .env file
load_dotenv()


# function to load openai model and geneerate response

def get_openai_response(question):
    llm_model = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),model_name="gpt-3.5-turbo-instruct", temperature=0.55)
    return llm_model(question)


# initializing streamlit app
st.set_page_config(page_title="OpenAI Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")
st.header("Langchain OpenAI Chatbot")
query_input = st.text_input("Enter your query here", key="query")
submit_button = st.button("Ask your query")

response = get_openai_response(query_input)
if submit_button:
    st.subheader("The response from OpenAI is:")
    st.write(response)
