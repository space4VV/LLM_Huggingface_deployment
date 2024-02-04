import streamlit as st
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os

# This loads all environment variables from .env file
load_dotenv()

# initializing streamlit app
st.set_page_config(
    page_title="OpenAI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)
cuisine_type = st.selectbox(
    "Select a cusine", ["Indian", "Chinese", "Italian", "Mexican", "American"]
)

def generate_restaurant_name_and_foods(cuisine_type):
    # Chain 1 : Restaurant name
    llm_model = OpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo-instruct",
        temperature=0.55,
    )
    prompt_rest_name = PromptTemplate(
        input_variables=["cuisine_type"],
        template="Suggest a cool name for the restaurant that serves {cuisine_type} food.",
    )
    rest_chain = LLMChain(llm=llm_model, prompt=prompt_rest_name,output_key="restaurant_name")

    prompt_food_name = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some cool food items for the restaurant {restaurant_name}.This should be the favourites of all!",

    )
    foods_chain = LLMChain(llm=llm_model, prompt=prompt_food_name, output_key="food_items")
    chain = SequentialChain(
        chains=[rest_chain, foods_chain],
        input_variables=["cuisine_type"],
        output_variables=["restaurant_name", "food_items"],
    )

    return chain.invoke({"cuisine_type": cuisine_type})

if cuisine_type:
    response = generate_restaurant_name_and_foods(cuisine_type)
    st.header(response["restaurant_name"].strip())
    food_items= response["food_items"].strip().split(",")
    for food_name in food_items:
        st.write("*",food_name)




if __name__ == "__main__":
    print(generate_restaurant_name_and_foods(cuisine_type))
