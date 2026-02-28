import streamlit as st
import pandas as pd
import pickle
import numpy as np
# ==============================
# background colours 
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#1565c0,#e3f2fd);
    color: white;
}

h1, h2, h3 {
    color: #1565c0;
}

div.stButton > button:first-child {
   
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)
# ==============================
# Load Models & Encoders
# ==============================

with open("models/workout_model.pkl", "rb") as f:
    workout_model = pickle.load(f)

with open("models/meal_model.pkl", "rb") as f:
    meal_model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

le_bmi = encoders["le_bmi"]
le_intensity = encoders["le_intensity"]
le_goal = encoders["le_goal"]
le_meal = encoders["le_meal"]

# ==============================
# Load Datasets
# ==============================

ex_df = pd.read_csv("Database/exercises.csv")
meal_df = pd.read_csv("Database/food.csv")

# Clean column names (very important)
ex_df.columns = ex_df.columns.str.strip()
meal_df.columns = meal_df.columns.str.strip()

# ==============================
# BMI Category Function
# ==============================

def bmi_category(weight, height):
    bmi = weight / (height ** 2)

    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# ==============================
# Streamlit UI
# ==============================

st.title(":rainbow-background[HYBRID AI FITNESS RECOMMENDER]")

weight = st.number_input("WEIGHT (kg)", min_value=30.0)
height = st.number_input("HEIGHT (meters)", min_value=1.0)

goal = st.selectbox("GOAL", ["Weight Loss", "Weight Gain", "Maintenance"])

bodypart = st.selectbox("TARGET BODY PART",["chest", "upper legs" , "upper arms", "lower legs", "lower arms", "shoulders", "back","waist", "neck"])

# ==============================
# When Button Clicked
# ==============================

if st.button("                  GET RECOMBENTATION             "):

    # -------- BMI Processing --------
    bmi_cat = bmi_category(weight, height)
    bmi_encoded = le_bmi.transform([bmi_cat])[0]
    goal_encoded = le_goal.transform([goal])[0]

    # -------- Workout Model Prediction --------
    intensity_pred = workout_model.predict([[bmi_encoded, goal_encoded]])
    intensity = le_intensity.inverse_transform(intensity_pred)[0]

    st.subheader("💪 Recommended Intensity Level")
    st.success(intensity)

    # -------- Exercise Filtering (Rule-Based) --------
    filtered_ex = ex_df[
        ex_df["bodyPart"].str.lower() == bodypart.lower()
    ]

    st.subheader("🏋️ Recommended Exercises")

    if filtered_ex.empty:
        st.write("DO CARDIO AND GO FOR WALK DAILY")
    else:
        for i, exercise in enumerate(filtered_ex["name"], 1):
            st.markdown(f"{i}. {exercise}")
        st.write("CHOOSE AND DO MINIMUM TWO TO MAXIMUM FIVE FROM THE LIST")

    # -------- Meal Model Prediction --------


    # -------- Meal Filtering (Rule-Based Logic) --------

    if goal == "Weight Loss":
        meal_filtered = meal_df.sort_values(by="Protein_g", ascending=False).head(5)

    elif goal == "Weight Gain":
        meal_filtered = meal_df.sort_values(by="Calories_per_100g", ascending=False).head(5)

    else:
        meal_filtered = meal_df.head(5)

    st.subheader("🥗 Recommended Meals")

    for i, meal in enumerate(meal_filtered["Food_Item"], 1):
        st.markdown(f"* {meal}")
    st.write("Always fill only half of your stomach and eat only one type of meal at a time")

    # -------- Disclaimer --------
    st.write("""
    :yellow-background[⚠ Disclaimer:
    This AI-generated recommendation is not medical advice.
    Please consult a certified trainer or healthcare professional
    before starting any diet or workout plan.]
    """)