import streamlit as st
import os
import google.generativeai as genai
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

# Set your Gemini API key here or use an environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'YOUR_GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="MyGPT", page_icon="🤖", layout="wide")

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
if "editing_chat" not in st.session_state:
    st.session_state.editing_chat = None

def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.chat_counter += 1
    chat_title = f"New Chat {st.session_state.chat_counter}"
    
    st.session_state.chats[chat_id] = {
        "title": chat_title,
        "messages": [
            {"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?"}
        ],
        "created_at": datetime.now()
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

def delete_chat(chat_id):
    """Delete a chat session"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chats:
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            else:
                st.session_state.current_chat_id = None

def generate_chat_title(user_input, chat_messages):
    """Generate an appropriate title for the chat based on user input and context"""
    try:
        # Create a context from recent messages
        recent_context = ""
        for msg in chat_messages[-3:]:  # Last 3 messages for context
            if msg["role"] == "user":
                recent_context += f"User: {msg['content']}\n"
        
        # Add the current user input
        context = recent_context + f"Current request: {user_input}"
        
        # Use AI to generate a title
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = f"""
        Based on this conversation context, generate a short, descriptive title (max 30 characters) for this chat.
        The title should capture the main topic or purpose of the conversation.
        
        Context:
        {context}
        
        Generate only the title, nothing else. Keep it concise and descriptive.
        """
        
        response = model.generate_content(prompt)
        title = response.text.strip()
        
        # Clean up the title
        title = title.replace('"', '').replace("'", "")
        if len(title) > 30:
            title = title[:27] + "..."
        
        return title if title else "Chat"
        
    except Exception as e:
        return "Chat"

def rename_chat(chat_id, new_title):
    """Rename a chat session"""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["title"] = new_title

def export_chat_to_pdf(chat_data):
    """Export chat to PDF"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20
    )
    
    user_style = ParagraphStyle(
        'UserMessage',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=10,
        backgroundColor='#e3f2fd'
    )
    
    assistant_style = ParagraphStyle(
        'AssistantMessage',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=10,
        backgroundColor='#f5f5f5'
    )
    
    story = []
    
    # Add title
    story.append(Paragraph(f"Chat: {chat_data['title']}", title_style))
    story.append(Paragraph(f"Created: {chat_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add messages
    for msg in chat_data["messages"]:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        content = f"{role_icon} {msg['role'].title()}: {msg['content']}"
        
        if msg["role"] == "user":
            story.append(Paragraph(content, user_style))
        else:
            story.append(Paragraph(content, assistant_style))
        
        story.append(Spacer(1, 10))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Sidebar for chat management
with st.sidebar:
    st.title("💬 Chats")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.divider()
    
    # Display existing chats
    if st.session_state.chats:
        for chat_id, chat_data in st.session_state.chats.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Chat title with click to select
                if st.button(
                    chat_data["title"], 
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    help="Click to select this chat"
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            
            with col2:
                # Edit button
                if st.button("✏️", key=f"edit_{chat_id}", help="Rename this chat"):
                    st.session_state.editing_chat = chat_id
                    st.rerun()
            
            with col3:
                # Delete button
                if st.button("🗑️", key=f"delete_{chat_id}", help="Delete this chat"):
                    delete_chat(chat_id)
                    st.rerun()
        
        # Edit chat title interface
        if st.session_state.editing_chat and st.session_state.editing_chat in st.session_state.chats:
            st.divider()
            st.subheader("✏️ Rename Chat")
            
            current_title = st.session_state.chats[st.session_state.editing_chat]["title"]
            new_title = st.text_input("New title:", value=current_title, key="edit_title_input")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Save", key="save_edit"):
                    if new_title.strip():
                        rename_chat(st.session_state.editing_chat, new_title.strip())
                        st.session_state.editing_chat = None
                        st.rerun()
            
            with col2:
                if st.button("❌ Cancel", key="cancel_edit"):
                    st.session_state.editing_chat = None
                    st.rerun()
    else:
        st.info("No chats yet. Create a new chat to get started!")

# Main chat area
if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Chat header
    st.header(f"🤖 {current_chat['title']}")
    
    # Display chat messages
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # User input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        current_chat["messages"].append({"role": "user", "content": user_input})
        
        # Auto-rename chat if it's still the default "New Chat X" title
        if current_chat["title"].startswith("New Chat") and len(current_chat["messages"]) <= 3:
            new_title = generate_chat_title(user_input, current_chat["messages"])
            if new_title and new_title != "Chat":
                current_chat["title"] = new_title
        
        # Prepare AI chat history format
        ai_history = []
        for msg in current_chat["messages"][:-1]:  # Exclude the last user message
            if msg["role"] == "user":
                ai_history.append({"role": "user", "parts": [msg["content"]]})
            else:
                ai_history.append({"role": "model", "parts": [msg["content"]]})
        
        # Use AI model for contextual understanding
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            chat = model.start_chat(history=ai_history)
            response = chat.send_message(user_input)
            assistant_reply = response.text
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)
            current_chat["messages"].append({"role": "assistant", "content": assistant_reply})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check your API key and model availability.")
    
    # Export functionality
    st.divider()
    st.subheader("Export Chat")
    
    if st.button("Export as PDF"):
        pdf_buffer = export_chat_to_pdf(current_chat)
        st.download_button(
            label="Download PDF",
            data=pdf_buffer.getvalue(),
            file_name=f"{current_chat['title']}.pdf",
            mime="application/pdf"
        )

else:
    # Welcome screen when no chat is selected
    st.title("Welcome to MyGPT")
    st.markdown("""
    ### Get Started
    1. Click **➕ New Chat** in the sidebar to start a conversation
    2. Type your message and press Enter
    3. Chat with AI with full conversation context
    
    ### Features
    - Multiple independent chat sessions
    - Persistent chat history
    - Modern chat interface
    - Powered by advanced AI
    - Auto-generated chat titles
    - Manual chat renaming
    - 📄 PDF export
    """)
    
    # Auto-create first chat if none exists
    if not st.session_state.chats:
        if st.button("Start Your First Chat"):
            create_new_chat()
            st.rerun() 