import streamlit as st
import os
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import tempfile
import nest_asyncio

# Allow nested event loops for Streamlit
nest_asyncio.apply()

# ----------------------------------------------------------------------------
# 🔑 Secure API Key Handling via Streamlit Secrets
# ----------------------------------------------------------------------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="MyGPT", page_icon="🤖", layout="wide")

# --- LangChain/LangGraph Setup ---
class ChatState(TypedDict):
    messages: List[BaseMessage]
    documents: Optional[List[str]]
    vectorstore: Optional[FAISS]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

MAX_HISTORY = 8

def prune_history(history):
    return history[-MAX_HISTORY:]

def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(splits, embeddings)
        os.unlink(tmp_file_path)
        return vectorstore, len(splits)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, 0

def create_rag_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

def chat_node(state: ChatState):
    pruned_history = prune_history(state["messages"])

    if state.get("vectorstore") and state.get("documents"):
        rag_chain = create_rag_chain(state["vectorstore"])

        latest_user_msg = next(
            (msg.content for msg in reversed(pruned_history) if isinstance(msg, HumanMessage)),
            None
        )

        if latest_user_msg:
            rag_response = rag_chain.invoke({"query": latest_user_msg})
            response_content = rag_response["result"]

            if rag_response.get("source_documents"):
                sources = [doc.metadata.get('page', 'Unknown') for doc in rag_response["source_documents"]]
                response_content += f"\n\n*Based on information from pages: {', '.join(map(str, sources))}*"
        else:
            response = llm.invoke(pruned_history)
            response_content = response.content
    else:
        response = llm.invoke(pruned_history)
        response_content = response.content

    state["messages"].append(AIMessage(content=response_content))
    return state

workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.set_finish_point("chat")
graph = workflow.compile()

# Streamlit Session State Initialization (unchanged)
# [ ... KEEP THE REST OF YOUR EXISTING SESSION STATE + UI CODE AS IS ... ]
