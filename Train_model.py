import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Load datasets
ex_df = pd.read_csv("Database/exercises.csv")
meal_df = pd.read_csv("Database/food.csv")   # Your nutrient dataset
def assign_intensity(row):
    equipment = str(row['equipment']).lower()

    if "barbell" in equipment or "machine" in equipment:
        return "Advanced"
    elif "dumbbell" in equipment:
        return "Intermediate"
    else:
        return "Beginner"

ex_df['Intensity'] = ex_df.apply(assign_intensity, axis=1)
le_bmi = LabelEncoder()
le_intensity = LabelEncoder()

# -----------------------------
# Create Logical Goal Mapping
# -----------------------------

def assign_goal_by_intensity(intensity):
    if intensity == "Beginner":
        return "Weight Loss"
    elif intensity == "Intermediate":
        return "Maintenance"
    else:
        return "Weight Gain"

ex_df['Goal'] = ex_df['Intensity'].apply(assign_goal_by_intensity)

# -----------------------------
# Encode Features
# -----------------------------

le_bmi = LabelEncoder()
le_intensity = LabelEncoder()
le_goal = LabelEncoder()

# Create realistic BMI categories (balanced)
bmi_categories = ["Underweight", "Normal", "Overweight", "Obese"]
ex_df['BMI_Category'] = np.random.choice(bmi_categories, len(ex_df))

ex_df['bmi_encoded'] = le_bmi.fit_transform(ex_df['BMI_Category'])
ex_df['goal_encoded'] = le_goal.fit_transform(ex_df['Goal'])
ex_df['intensity_encoded'] = le_intensity.fit_transform(ex_df['Intensity'])

# -----------------------------
# Train Workout Model
# -----------------------------

X_workout = ex_df[['bmi_encoded', 'goal_encoded']]
y_workout = ex_df['intensity_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X_workout, y_workout, test_size=0.2, random_state=42
)

workout_model = RandomForestClassifier(class_weight="balanced")
workout_model.fit(X_train, y_train)

print("Workout Model Report")
print(classification_report(y_test, workout_model.predict(X_test)))
def meal_category(row):
    if row['Protein_g'] > row['Carbs_g'] and row['Calories_per_100g'] < 400:
        return "Weight Loss"
    elif row['Calories_per_100g'] > 500:
        return "Weight Gain"
    else:
        return "Maintenance"

meal_df['Meal_Category'] = meal_df.apply(meal_category, axis=1)
le_meal = LabelEncoder()

meal_df['meal_encoded'] = le_meal.fit_transform(meal_df['Meal_Category'])
X_meal = meal_df[['Protein_g', 'Carbs_g', 'Calories_per_100g']]
y_meal = meal_df['meal_encoded']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_meal, y_meal, test_size=0.2, random_state=42
)

meal_model = RandomForestClassifier()
meal_model.fit(X_train_m, y_train_m)

print("Meal Model Report")
print(classification_report(y_test_m, meal_model.predict(X_test_m)))
if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(workout_model, open("models/workout_model.pkl", "wb"))
pickle.dump(meal_model, open("models/meal_model.pkl", "wb"))

pickle.dump({
    "le_bmi": le_bmi,
    "le_intensity": le_intensity,
    "le_goal": le_goal,
    "le_meal": le_meal
}, open("models/encoders.pkl", "wb"))

print("Models saved successfully!")
