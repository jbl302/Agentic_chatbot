import streamlit as st
import requests

st.set_page_config(page_title='Agents using langgraph',layout="wide")
st.title('AI chatbot agents')
st.write('interact with agents')

system_prompt = st.text_area('define the role of AI agent', height = 70, placeholder='Define the type of agent')

groq_models = ['qwen-qwq-32b', 'deepseek-r1-distill-qwen-32b',
                       'llama-3.3-70b-specdec', 'llama-3.2-3b-preview']
deepseek_models = ['deepseek-chat']

provider = st.radio("select provider",{'groq','deepseek'})

if provider == 'groq':
  select_box = st.selectbox('select among these',groq_models)
else:
  select_box = st.selectbox('deepseek model',deepseek_models)
  
allowe_web_search = st.checkbox('allow web search')

user_query = st.text_area('Ask me anything haha', height = 70, placeholder='write query here')


API_URL = 'http://127.0.0.1:8000/chat'
if st.button('Ask agent'):
  if user_query.strip():
    
    payload = {
      "model_name": select_box,
      "model_provider": provider,
      "system_prompt": system_prompt,
      # Ensure each message is a dictionary with "role" and "content"
      "messages": user_query,

      "allow_search": allowe_web_search
    }
    st.write(allowe_web_search)
    response  = requests.post(API_URL,json=payload)
    if response.status_code == 200:
      
      response_data = response.json()
      if "error" in response_data:
        st.error(response_data['error'])    
    
      st.subheader("agent response")
      st.markdown(f"Final response {response_data}")