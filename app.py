import streamlit as st
import pickle
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os
import hashlib
from datetime import datetime, timedelta
import json

# Security and configuration
st.set_page_config(
    page_title="Marathon Training Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Secure state management
def init_session_state():
    INITIAL_STATES = {
        'chat_history': [],
        'coach': None,
        'embeddings_done': False,
        'training_plan': None,
        'runs': [],
        'last_embedding_time': None
    }
    for key, value in INITIAL_STATES.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Security warning
def show_security_guidelines():
    with st.sidebar.expander("📚 Security Guidelines", expanded=False):
        st.markdown("""
        ### Security Best Practices
        1. Never share your OpenAI API key
        2. Don't upload sensitive personal data
        3. Clear your training data when done
        4. Use the 'Clear All Data' button below when needed
        """)

# Data management
def clear_session_data():
    """Securely clear all session data"""
    # Clear temporary files
    temp_files = [f for f in os.listdir('.') if f.endswith('.pkl') or f.endswith('.epub')]
    for f in temp_files:
        try:
            os.remove(f)
        except:
            pass
    
    # Reset session state
    st.session_state.chat_history = []
    st.session_state.coach = None
    st.session_state.embeddings_done = False
    st.session_state.training_plan = None
    st.session_state.runs = []
    st.session_state.last_embedding_time = None
    
    st.sidebar.success("All data cleared!")

# Secure file handling
def secure_file_hash(file_bytes):
    """Generate a secure hash of the file"""
    return hashlib.blake2b(file_bytes, digest_size=16).hexdigest()

# Main interface
def main():
    show_security_guidelines()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Setup")
        
        # API Key handling
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. It will not be stored."
        )
        
        # File upload with size check
        uploaded_file = st.file_uploader(
            "Upload training book (ePub)",
            type=['epub'],
            help="Upload your training book (max 200MB)"
        )
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            clear_session_data()
    
    # Process book and create coach if conditions are met
    if uploaded_file and openai_api_key:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > 200 * 1024 * 1024:  # 200MB limit
            st.error("File too large. Please upload a smaller file.")
            return
        
        file_hash = secure_file_hash(file_bytes)
        embeddings_path = f"embeddings_{file_hash}.pkl"
        
        # Handle embeddings
        try:
            if process_embeddings(file_bytes, file_hash, openai_api_key, embeddings_path):
                display_main_interface()
        except Exception as e:
            st.error(f"Error processing book: {str(e)}")
            clear_session_data()
    else:
        st.write("👋 Welcome to Marathon Training Coach!")
        st.write("Please provide your OpenAI API key and upload your training book to get started.")

def process_embeddings(file_bytes, file_hash, api_key, embeddings_path):
    """Process and store embeddings securely"""
    if os.path.exists(embeddings_path) and st.session_state.embeddings_done:
        return True
        
    with st.spinner("Processing your training book..."):
        try:
            # Save temporary file
            with open("temp_book.epub", "wb") as f:
                f.write(file_bytes)
            
            # Set API key for this session
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Process book
            loader = UnstructuredEPubLoader("temp_book.epub")
            documents = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings
            embedding = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding
            )
            
            # Setup QA chain
            st.session_state.coach = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.7, model_name="gpt-4"),
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            st.session_state.embeddings_done = True
            st.session_state.last_embedding_time = datetime.now()
            
            # Clean up
            if os.path.exists("temp_book.epub"):
                os.remove("temp_book.epub")
                
            return True
            
        except Exception as e:
            st.error(f"Error processing book: {str(e)}")
            clear_session_data()
            return False

def display_main_interface():
    """Display the main app interface"""
    tab1, tab2, tab3 = st.tabs(["📋 Training Plan", "📝 Log Runs", "💬 Coach Chat"])
    
    with tab1:
        display_training_plan_generator()
    
    with tab2:
        display_run_logger()
    
    with tab3:
        display_coach_chat()

def display_training_plan_generator():
    st.header("Generate Your Training Plan")
    
    col1, col2 = st.columns(2)
    with col1:
        weeks = st.selectbox("Plan Length (weeks)", options=[12, 14, 16, 18, 20, 24])
        peak_miles = st.number_input("Peak Weekly Mileage", 20, 120, 50)
        current_miles = st.number_input("Current Weekly Mileage", 0, 100, 20)
    
    with col2:
        race_date = st.date_input(
            "Race Date",
            min_value=datetime.now().date() + timedelta(weeks=8)
        )
        experience = st.selectbox(
            "Experience Level",
            ["Beginner", "Intermediate", "Advanced"]
        )
    
    if st.button("Generate Plan"):
        generate_training_plan(weeks, peak_miles, current_miles, race_date, experience)

# Add the rest of your interface functions here...

if __name__ == "__main__":
    main()
