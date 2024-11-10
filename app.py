import streamlit as st
import pickle
from langchain.document_loaders import TextLoader  # Changed this line
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # Changed this line
from langchain.embeddings import OpenAIEmbeddings  # Changed this line
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os
import hashlib
from datetime import datetime, timedelta

# Security and configuration
st.set_page_config(
    page_title="Marathon Training Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'coach' not in st.session_state:
    st.session_state.coach = None
if 'embeddings_done' not in st.session_state:
    st.session_state.embeddings_done = False
if 'training_plan' not in st.session_state:
    st.session_state.training_plan = None

# Main interface
with st.sidebar:
    st.title("⚙️ Setup")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload training plan (txt)", type=['txt'])
    
    if uploaded_file and openai_api_key:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Save temporary file
        with open("temp_plan.txt", "wb") as f:
            f.write(file_bytes)
        
        # Set up OpenAI key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # Process the document
        if not st.session_state.embeddings_done:
            with st.spinner("Processing your training document..."):
                # Load and split the text
                loader = TextLoader("temp_plan.txt")
                documents = loader.load()
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
                
                # Create the conversational chain
                st.session_state.coach = ConversationalRetrievalChain.from_llm(
                    ChatOpenAI(temperature=0.7, model_name="gpt-4"),
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True
                )
                
                st.session_state.embeddings_done = True
                st.success("Ready to help with your training!")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📋 Training Plan", "📝 Log Runs", "💬 Coach Chat"])

with tab1:
    st.header("Generate Your Training Plan")
    col1, col2 = st.columns(2)
    
    with col1:
        weeks = st.selectbox("Plan Length (weeks)", options=[12, 14, 16, 18, 20, 24])
        peak_miles = st.number_input("Peak Weekly Mileage", 20, 120, 50)
    
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
        if st.session_state.coach:
            query = f"""Create a {weeks}-week marathon training plan for an {experience} runner targeting {peak_miles} peak weekly mileage, with race date {race_date}. Include weekly mileage, long run distances, and key workouts."""
            
            response = st.session_state.coach({"question": query, "chat_history": []})
            st.session_state.training_plan = response["answer"]
            st.write(response["answer"])
            
            # Add download button
            st.download_button(
                "Download Plan",
                response["answer"],
                file_name="marathon_training_plan.txt"
            )

with tab2:
    st.header("Log Your Run")
    distance = st.number_input("Distance (miles)", 0.0, 50.0, 5.0)
    duration = st.text_input("Duration (HH:MM:SS)", "00:45:00")
    feel = st.selectbox("How did it feel?", ["Easy", "Medium", "Hard"])
    
    if st.button("Log Run"):
        if st.session_state.coach:
            query = f"Give feedback on this run: {distance} miles in {duration}, felt {feel}"
            response = st.session_state.coach({"question": query, "chat_history": st.session_state.chat_history})
            st.write(response["answer"])

with tab3:
    st.header("Ask Your Coach")
    question = st.text_input("Ask a training question:")
    
    if question and st.session_state.coach:
        response = st.session_state.coach({"question": question, "chat_history": st.session_state.chat_history})
        st.write(response["answer"])
