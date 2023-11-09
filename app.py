import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
import langchain
langchain.verbose = False
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
  
#new section
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')



def get_youtube_transcript(url):
    try:
        # Extract the video ID from the URL
        video_id = url.split('v=')[1].split('&')[0]

        # Retrieve the available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Find and store the English transcript
        english_transcript = None
        for transcript in transcript_list:
            if transcript.language_code == 'en':
                english_transcript = transcript.fetch()
                break  # Exit the loop after finding the English transcript

        if english_transcript is not None:
            # Join the list of transcript segments into a single string
            english_transcript_text = '\n'.join(segment['text'] for segment in english_transcript)
            return english_transcript_text
        else:
            return "English transcript not found for the video."
    except IndexError:
        return "Invalid YouTube URL format. Please provide a valid URL."

# Example usage:
# youtube_url = 'https://www.youtube.com/watch?v=AkA58hX4ZDc&ab_channel=DavidMbugua'
# transcript = get_youtube_transcript(youtube_url)
# print(transcript)

def webscrap(name):
    # Replace this URL with the one you want to scrape
    url = f'https://www.{name}.com'

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()
        return page_text
    else:
        return None

#youtube transcript
# def youtube( link ):
    
st.title("SCRAPPER AI")
# Create a radio button to choose the chat mode
chat_mode = st.radio("Select Chat Mode:", ("YOUTUBE TRANSCRIPTS", "WEB SCRAPPER"))
# Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


if chat_mode == "YOUTUBE TRANSCRIPTS":
    
    url = st.text_input("enter video url")
    
    web_data= get_youtube_transcript(url)
    if web_data is not None:
        
        text = web_data
        # for page in pdf_reader.pages:
        #     text += page.extract_text()


        max_length = 1800
        original_string = text
        temp_string = ""
        strings_list = []
       #split into chunks
        for character in original_string:
            if len(temp_string) < max_length:
                temp_string += character
            else:
                strings_list.append(temp_string)
                temp_string = ""

        if temp_string:
            strings_list.append(temp_string)

      
        

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(strings_list, embedding=embeddings)

        user_question = st.text_input("Ask question regarding the video")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.9)

            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = user_question)
                print(cb)

            st.write(response)

        


elif chat_mode == "WEB SCRAPPER":
    
    name = st.text_input("enter website name")
    
    web_data= webscrap(name)
    if web_data is not None:
        
        text = web_data
        # for page in pdf_reader.pages:
        #     text += page.extract_text()


        max_length = 1500
        original_string = text
        temp_string = ""
        strings_list = []

        for character in original_string:
            if len(temp_string) < max_length:
                temp_string += character
            else:
                strings_list.append(temp_string)
                temp_string = ""

        if temp_string:
            strings_list.append(temp_string)

        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(strings_list, embedding=embeddings)

        user_question = st.text_input("Ask question regarding the website data")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.9)

            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = user_question)
                print(cb)

            st.write(response)