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
nest_asyncio.apply()


GOOGLE_API_KEY = "AIzaSyD4oSWK6Djn2Gcvte06dogN1iJkm7eOpzE"
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

st.set_page_config(page_title="MyGPT", page_icon="🤖", layout="wide")

# --- LangChain/LangGraph Setup ---
class ChatState(TypedDict):
    messages: List[BaseMessage]
    documents: Optional[List[str]]  # Store document names
    vectorstore: Optional[FAISS]    # Store vector store for RAG

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
MAX_HISTORY = 8  # 4 rounds (user-AI-user-AI...)

def prune_history(history):
    return history[-MAX_HISTORY:]

def process_pdf(uploaded_file):
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and split PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(pages)
        
        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vectorstore, len(splits)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, 0

def create_rag_chain(vectorstore):
    """Create RAG chain for document retrieval"""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

def chat_node(state: ChatState):
    pruned_history = prune_history(state["messages"])
    
    # Check if we have documents for RAG
    if state.get("vectorstore") and state.get("documents"):
        # Use RAG for context-aware responses
        rag_chain = create_rag_chain(state["vectorstore"])
        
        # Get the latest user message
        latest_user_msg = None
        for msg in reversed(pruned_history):
            if isinstance(msg, HumanMessage):
                latest_user_msg = msg.content
                break
        
        if latest_user_msg:
            # Get RAG response
            rag_response = rag_chain.invoke({"query": latest_user_msg})
            response_content = rag_response["result"]
            
            # Add context about sources if available
            if rag_response.get("source_documents"):
                sources = [doc.metadata.get('page', 'Unknown') for doc in rag_response["source_documents"]]
                response_content += f"\n\n*Based on information from pages: {', '.join(map(str, sources))}*"
        else:
            response = llm.invoke(pruned_history)
            response_content = response.content
    else:
        # Regular chat without RAG
        response = llm.invoke(pruned_history)
        response_content = response.content
    
    # Add AIMessage to state
    state["messages"].append(AIMessage(content=response_content))
    return state


workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.set_finish_point("chat")
graph = workflow.compile()

# --- Streamlit Session State ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
if "editing_chat" not in st.session_state:
    st.session_state.editing_chat = None

# --- PDF Export ---
def export_chat_to_pdf(chat_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=20)
    user_style = ParagraphStyle('UserMessage', parent=styles['Normal'], fontSize=10, leftIndent=20, spaceAfter=10, backgroundColor='#e3f2fd')
    assistant_style = ParagraphStyle('AssistantMessage', parent=styles['Normal'], fontSize=10, leftIndent=20, spaceAfter=10, backgroundColor='#f5f5f5')
    story = []
    story.append(Paragraph(f"Chat: {chat_data['title']}", title_style))
    story.append(Paragraph(f"Created: {chat_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    for msg in chat_data["messages"]:
        role_icon = "👤" if isinstance(msg, HumanMessage) else "🤖"
        content = f"{role_icon} {msg.content}"
        if isinstance(msg, HumanMessage):
            story.append(Paragraph(content, user_style))
        else:
            story.append(Paragraph(content, assistant_style))
        story.append(Spacer(1, 10))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Chat Management Functions ---
def create_new_chat(initial_title=None):
    chat_id = str(uuid.uuid4())
    st.session_state.chat_counter += 1
    chat_title = initial_title if initial_title else f"New Chat {st.session_state.chat_counter}"
    memory = ConversationBufferMemory(return_messages=True)
    # Add the assistant's greeting to memory
    memory.chat_memory.add_ai_message("Hello! I'm your AI assistant. How can I help you today?")
    st.session_state.chats[chat_id] = {
        "title": chat_title,
        "memory": memory,
        "created_at": datetime.now(),
        "documents": [],  # Store uploaded document names
        "vectorstore": None  # Store vector store for RAG
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chats:
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            else:
                st.session_state.current_chat_id = None

def rename_chat(chat_id, new_title):
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["title"] = new_title

# --- Sidebar ---
with st.sidebar:
    st.title("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    # PDF Upload Section
    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload a PDF document for RAG",
        type=['pdf'],
        help="Upload a PDF to enable context-aware responses based on the document content"
    )
    
    if uploaded_file is not None and st.session_state.current_chat_id:
        if st.button("🔍 Process PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                vectorstore, num_chunks = process_pdf(uploaded_file)
                if vectorstore:
                    current_chat = st.session_state.chats[st.session_state.current_chat_id]
                    current_chat["vectorstore"] = vectorstore
                    current_chat["documents"].append(uploaded_file.name)
                    st.success(f"✅ PDF processed successfully! Created {num_chunks} text chunks.")
                    st.rerun()
    
    # Show uploaded documents
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        if current_chat["documents"]:
            st.divider()
            st.subheader("📚 Loaded Documents")
            for doc in current_chat["documents"]:
                st.write(f"📄 {doc}")
            if st.button("Clear Documents", use_container_width=True):
                current_chat["documents"] = []
                current_chat["vectorstore"] = None
                st.success("Documents cleared!")
                st.rerun()
    
    st.divider()
    if st.session_state.chats:
        for chat_id, chat_data in st.session_state.chats.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                if st.button(chat_data["title"], key=f"chat_{chat_id}", use_container_width=True, help="Click to select this chat"):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"edit_{chat_id}", help="Rename this chat"):
                    st.session_state.editing_chat = chat_id
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{chat_id}", help="Delete this chat"):
                    delete_chat(chat_id)
                    st.rerun()
        if st.session_state.editing_chat and st.session_state.editing_chat in st.session_state.chats:
            st.divider()
            st.subheader("✏️ Rename Chat")
            current_title = st.session_state.chats[st.session_state.editing_chat]["title"]
            new_title = st.text_input("New title:", value=current_title, key="edit_title_input")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Save", key="save_edit"):
                    if new_title and new_title.strip():
                        rename_chat(st.session_state.editing_chat, new_title.strip())
                        st.session_state.editing_chat = None
                        st.rerun()
            with col2:
                if st.button("❌ Cancel", key="cancel_edit"):
                    st.session_state.editing_chat = None
                    st.rerun()
    else:
        st.info("No chats yet. Create a new chat to get started!")

# --- Main Chat Area ---
if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    memory = current_chat["memory"]
    
    # Show RAG status
    rag_status = "🟢 RAG Enabled" if current_chat["vectorstore"] else "🔴 RAG Disabled"
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.header(f"🤖 {current_chat['title']}")
    with col2:
        st.write("")
        st.write("")
        st.info(rag_status)
    with col3:
        st.write("")
        st.write("")
    with col4:
        st.write("")
        st.write("")
        if st.button("📄 Export PDF", help="Export this chat as PDF"):
            pdf_buffer = export_chat_to_pdf({
                "title": current_chat["title"],
                "created_at": current_chat["created_at"],
                "messages": memory.chat_memory.messages
            })
            st.download_button(
                label="Download PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"{current_chat['title']}.pdf",
                mime="application/pdf"
            )
    
    # Display chat messages
    for msg in memory.chat_memory.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
    
    # User input
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Set chat title if it's the first message
        if len(memory.chat_memory.messages) == 1 and current_chat["title"].startswith("New Chat"):
            current_chat["title"] = user_input.strip()[:40]

        with st.chat_message("user"):
            st.markdown(user_input)

        # Add user's message to memory
        memory.chat_memory.add_user_message(user_input)

        # Run through LangGraph for memory and response
        state = ChatState(
            messages=memory.chat_memory.messages,
            documents=current_chat["documents"],
            vectorstore=current_chat["vectorstore"]
        )
        result = graph.invoke(state)

        # Overwrite memory with updated state
        memory.chat_memory.messages = result["messages"]

        # Display latest assistant message
        ai_msg = state["messages"][-1]
        with st.chat_message("assistant"):
            st.markdown(ai_msg.content)

else:
    st.title("Welcome to MyGPT")
    st.markdown("""
    ### Get Started
    1. Click **➕ New Chat** in the sidebar to start a conversation
    2. Upload a PDF document for context-aware responses (optional)
    3. Type your message and press Enter
    4. Chat with AI with full conversation context and document knowledge
    
    ### Features
    - Multiple independent chat sessions
    - Persistent chat history
    - Modern chat interface
    - Powered by advanced AI (LangChain + LangGraph)
    - ** PDF Document Upload & RAG** - Upload PDFs for context-aware responses
    - Auto-generated chat titles
    - Manual chat renaming
    -  PDF export
    """)
    if not st.session_state.chats:
        if st.button("Start Your First Chat"):
            create_new_chat()
            st.rerun() 