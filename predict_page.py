import streamlit as st
import pickle
import numpy as np



def load_model():
    with open('algorithm.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

classification = data["model"]


def labelencoder_load():
    with open('encoder.pkl', 'rb') as file:
        le = pickle.load(file)
    return le

label = labelencoder_load()

le = label["labelencoder"]


def show_predict_page():
    # Title
    st.title("Student Depression Prediction")

    # Sub-title
    st.write("""### Please fill in the details below""")

    # Dropdown options
    Degree_list = (
        'B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'PhD', 'Class 12', 'B.Ed',
        'LLB', 'BE', 'M.Ed', 'MSc', 'BHM', 'M.Pharm', 'MCA', 'MA', 'B.Com',
        'MD', 'MBA', 'MBBS', 'M.Com', 'B.Arch', 'LLM', 'B.Tech', 'BBA',
        'ME', 'MHM', 'Others'
    )

    # User input
    Age = st.slider("Select your age", 0, 100, 18)
    Gender = st.selectbox("Select gender", ["Male", "Female"])
    Academic_Pressure = st.number_input("Enter Academic Pressure", min_value=0.0, max_value=5.0, value=0.0)
    CGPA = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=0.0)
    Study_Satisfaction = st.number_input("Enter Study Satisfaction", min_value=0.0, max_value=5.0, value=0.0)
    Sleep_Duration = st.number_input("Enter Sleep Duration", min_value=3.0, max_value=8.0, value=3.0)
    Dietary_Habits = st.selectbox("Enter Dietary Habits", ['Unhealthy', 'Moderate', 'Healthy'])
    Degree = st.selectbox("Enter Degree", Degree_list)
    Have_you_ever_had_suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ['Yes', 'No'])
    Study_Hours = st.number_input("Enter Study Hours", min_value=1.0, max_value=12.0, value=1.0)
    Financial_Stress = st.number_input("Enter Financial Stress", min_value=0.0, max_value=5.0, value=0.0)
    Family_History_of_Mental_Illness = st.selectbox("Family History of Mental Illness", ['Yes', 'No'])

    # Gender and Dietary Habits mapping
    gender_mapping = {"Male": 1, "Female": 0}
    Dietary_Habits_mapping = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
    sucide_thought_mapping = {'Yes': 1, 'No': 0}
    family_history_mapping = {'Yes': 1, 'No': 0}
    Degree_map = {
        'B.Pharm':0, 'BSc':1, 'BA':2, 'BCA':3, 'M.Tech':4, 'PhD':5, 'Class 12':6, 'B.Ed':7,
        'LLB':8, 'BE':9, 'M.Ed':10, 'MSc':11, 'BHM':12, 'M.Pharm':13, 'MCA':14, 'MA':15, 'B.Com':16,
        'MD':17, 'MBA':18, 'MBBS':19, 'M.Com':20, 'B.Arch':21, 'LLM':22, 'B.Tech':23, 'BBA':24,
        'ME':25, 'MHM':26, 'Others':27}
    


    # Convert inputs to numeric
    gender_numeric = gender_mapping[Gender]
    Dietary_Habits_numeric = Dietary_Habits_mapping[Dietary_Habits]
    sucide_thought_numeric = sucide_thought_mapping[Have_you_ever_had_suicidal_thoughts]
    family_history_numeric = family_history_mapping[Family_History_of_Mental_Illness]
    Degree_numeric = Degree_map[Degree]

    # Button to predict
    ok = st.button("Submit")
    if ok:        
        # Create input array
        x = np.array([[Age, gender_numeric, Academic_Pressure, CGPA, Study_Satisfaction, 
                       Sleep_Duration, Dietary_Habits_numeric, Degree_numeric, 
                       sucide_thought_numeric, Study_Hours, Financial_Stress, 
                       family_history_numeric]])
        

        
        # Predict using the loaded model
        try:
            patient = classification.predict(x)
            result_mapping = {1: "Student Needs Attention", 0: "Student is Healthy"}
            result_label = result_mapping.get(int(patient[0]), "Unknown result")
            st.subheader(f"Prediction: {result_label}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


