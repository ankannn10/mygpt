# Cell 1
# %pip install google-genai

# Cell 2
import os
import google.generativeai as genai

os.environ['GOOGLE_API_KEY'] = "AIzaSyB6A7Dv4XAfUA7cfgSItclr6Rk3OrxLQ7o"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel('gemini-1.5-flash')
'''
response = model.generate_content("List 5 planets each with an interesting fact")
print(response.text)

response = model.generate_content("what are top 5 frequently used emojis?")
print(response.text)

# Cell 3
# safeguarding the responses
response = model.generate_content("How can I hack into someone's email account?")
print(response.text)
print(response.prompt_feedback)

# Cell 4
response = model.generate_content("What is Quantum Computing?",
                                  generation_config = genai.types.GenerationConfig(
                                      candidate_count = 1,
                                      stop_sequences = ['.'],
                                      max_output_tokens = 40,
                                      top_p = 0.6,
                                      top_k = 5,
                                      temperature = 0.8)
                                 )
print(response.text)

# Cell 5
import PIL

image = PIL.Image.open('/Users/ankanchakraborty/Documents/mygpt/images.jpeg')
vision_model = genai.GenerativeModel('gemini-1.5-flash')
response = vision_model.generate_content(["Explain the picture?",image])
print(response.text)

# Cell 6
chat_model = genai.GenerativeModel('gemini-1.5-flash')
chat = chat_model .start_chat(history=[])

response = chat.send_message("Which is one of the best place to visit in India during summer?")
print(response.text)
response = chat.send_message("Tell me more about that place in 50 words")
print(response.text)
print(chat.history)
'''
# Cell 7
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
response = llm.invoke("Explain Quantum Computing in 50 words")
print(response.content)

# Cell 8
batch_responses = llm.batch(
    [
        "What is the capital of Finland?",
        "Who is the Chairman of Wipro?",
    ]
)
for response in batch_responses:
    print(response.content)

# Cell 9
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe the image",
        },
        {
            "type": "image_url",
            "image_url": "https://picsum.photos/id/237/200/300"
        },
    ]
)

response = llm.invoke([message])
print(response.content)

# Cell 10
# !pip install streamlit

# Cell 11
import streamlit as st
import os
import google.generativeai as genai

st.title("Gemini Bot")

os.environ['GOOGLE_API_KEY'] = "AIzaSyB6A7Dv4XAfUA7cfgSItclr6Rk3OrxLQ7o"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

# Select the model
model = genai.GenerativeModel('gemini-pro')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":"assistant",
            "content":"Ask me Anything"
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process and store Query and Response
def llm_function(query):
    response = model.generate_content(query)

    # Displaying the Assistant Message
    with st.chat_message("assistant"):
        st.markdown(response.text)

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content": query
        }
    )

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content": response.text
        }
    )

# Accept user input
query = st.chat_input("What's up?")

# Calling the Function when Input is Provided
if query:
    # Displaying the User Message
    with st.chat_message("user"):
        st.markdown(query)

    llm_function(query)
