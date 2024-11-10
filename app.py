import streamlit as st
import pickle
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
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
if 'training_principles' not in st.session_state:
    st.session_state.training_principles = None

# Main interface
with st.sidebar:
    st.title("⚙️ Setup")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload training book (txt)", type=['txt'])
    
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
                
                # Create embeddings and vectorstore
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                # Create the conversational chain
                st.session_state.coach = ConversationalRetrievalChain.from_llm(
                    ChatOpenAI(temperature=0.7, model_name="gpt-4"),
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True
                )
                
                st.session_state.embeddings_done = True
                st.success("Ready to help with your training!")

# Create the main tabs
tab1, tab2, tab3 = st.tabs(["📋 Training Plan", "📝 Log Runs", "💬 Coach Chat"])

# Training Plan Tab
with tab1:
    st.header("Generate Your Training Plan")
    
    # Training Preferences
    st.subheader("Your Details")
    col1, col2 = st.columns(2)
    
    with col1:
        weeks = st.selectbox("Plan Length (weeks)", options=[12, 14, 16, 18, 20, 24])
        peak_miles = st.number_input("Peak Weekly Mileage Target", 20, 120, 50)
        current_miles = st.number_input("Current Weekly Mileage", 0, 100, 20)
    
    with col2:
        race_date = st.date_input(
            "Race Date",
            min_value=datetime.now().date() + timedelta(weeks=8)
        )
        experience = st.selectbox(
            "Running Experience",
            ["Beginner", "Intermediate", "Advanced"]
        )
        runs_per_week = st.slider("Preferred runs per week", 3, 7, 5)
    
    # Additional Training Preferences
    st.subheader("Training Preferences")
    col3, col4 = st.columns(2)
    
    with col3:
        long_run_day = st.selectbox("Preferred long run day", 
            ["Saturday", "Sunday"], 
            index=0
        )
        cross_training = st.checkbox("Include cross-training recommendations", value=True)
        
    with col4:
        goal_type = st.radio(
            "Primary Goal",
            ["Finish comfortably", "Target time", "Personal best"]
        )
        if goal_type == "Target time":
            target_time = st.text_input("Target finish time (HH:MM)", "04:00")
    
    if st.button("Generate Personalized Plan"):
        if st.session_state.coach:
            # First, analyze the book's training principles
            principles_query = """Based on the running book provided, what are the key training principles for marathon preparation? 
            Specifically analyze:
            1. How does the book approach building weekly mileage?
            2. What is the philosophy on long runs and their progression?
            3. What types of workouts (tempo, intervals, etc.) does it recommend?
            4. How does it handle recovery and rest days?
            5. What is the approach to tapering?
            Provide a concise summary of these principles."""
            
            principles_response = st.session_state.coach(
                {"question": principles_query, "chat_history": []}
            )
            
            # Create the training plan based on these principles
            plan_context = f"""Using the training principles you just analyzed, create a {weeks}-week marathon training plan with these specifications:

            Runner Profile:
            - Experience Level: {experience}
            - Current weekly mileage: {current_miles}
            - Target peak mileage: {peak_miles}
            - Preferred running days: {runs_per_week} days per week
            - Long run day: {long_run_day}
            - Goal: {goal_type} {"(" + target_time + ")" if goal_type == "Target time" else ""}
            
            Please provide:
            1. A week-by-week plan showing:
               - Weekly mileage target
               - Specific workouts for each day
               - Long run distances
               - Recovery days
               {"- Cross-training recommendations" if cross_training else ""}
            2. Pace guidelines for different types of runs
            3. Key milestones and checkpoints
            4. Guidelines for adjusting the plan based on fatigue or missed workouts
            
            Format the plan clearly with week numbers and daily details.
            """
            
            with st.spinner("Creating your personalized training plan..."):
                # Show the principles being applied
                st.info("📚 Training Principles from the Book")
                st.write(principles_response["answer"])
                
                # Generate the plan
                response = st.session_state.coach(
                    {"question": plan_context, "chat_history": []}
                )
                st.session_state.training_plan = response["answer"]
                
                st.success("🎯 Your Personalized Training Plan")
                st.write(response["answer"])
                
                # Add download options
                st.download_button(
                    "📥 Download Training Plan",
                    response["answer"],
                    file_name=f"marathon_training_plan_{race_date.strftime('%Y-%m-%d')}.txt",
                    help="Download your plan as a text file"
                )
        else:
            st.error("Please set up your coach first by uploading your training book and adding your OpenAI API key")

# Run Logging Tab
with tab2:
    st.header("Log Your Run")
    distance = st.number_input("Distance (miles)", 0.0, 50.0, 5.0)
    duration = st.text_input("Duration (HH:MM:SS)", "00:45:00")
    feel = st.selectbox("How did it feel?", ["Easy", "Medium", "Hard"])
    
    notes = st.text_area("Notes (optional)", placeholder="How was your run? Any issues?")
    
    if st.button("Log Run"):
        if st.session_state.coach:
            query = f"""Give feedback on this run based on the training principles in the book:
            - Distance: {distance} miles
            - Duration: {duration}
            - Effort level: {feel}
            - Runner's notes: {notes}
            
            Please provide:
            1. Analysis of the run
            2. Recommendations for recovery
            3. Suggestions for the next workout
            """
            response = st.session_state.coach({"question": query, "chat_history": st.session_state.chat_history})
            st.write(response["answer"])

# Coach Chat Tab
with tab3:
    st.header("Ask Your Coach")
    question = st.text_input("Ask a training question:")
    
    if question and st.session_state.coach:
        response = st.session_state.coach({"question": question, "chat_history": st.session_state.chat_history})
        st.write(response["answer"])
