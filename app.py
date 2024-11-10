# In the tab1 section, replace the existing code with this:

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

# Add this to the top of your file with other session state initializations
if 'training_principles' not in st.session_state:
    st.session_state.training_principles = None
