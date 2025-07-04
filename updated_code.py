import pandas as pd
import streamlit as st
import joblib
import os

# Load the trained model
model = joblib.load("model.pkl")

# Feature list (same order as in training)
features = ['sudden_fever', 'headache', 'mouth_bleed', 'nose_bleed', 'muscle_pain',
    'joint_pain', 'vomiting', 'rash', 'diarrhea', 'hypotension',
    'pleural_effusion', 'ascites', 'gastro_bleeding', 'swelling', 'nausea',
    'chills', 'myalgia', 'digestion_trouble', 'fatigue', 'skin_lesions',
    'stomach_pain', 'orbital_pain', 'neck_pain', 'weakness', 'back_pain',
    'weight_loss', 'gum_bleed', 'jaundice', 'coma', 'diziness',
    'inflammation', 'red_eyes', 'loss_of_appetite', 'urination_loss',
    'slow_heart_rate', 'abdominal_pain', 'light_sensitivity', 'yellow_skin',
    'yellow_eyes', 'facial_distortion', 'microcephaly', 'rigor',
    'bitter_tongue', 'convulsion', 'anemia', 'cocacola_urine',
    'hypoglycemia', 'prostraction', 'hyperpyrexia', 'stiff_neck',
    'irritability', 'confusion', 'tremor', 'paralysis', 'lymph_swells',
    'breathing_restriction', 'toe_inflammation', 'finger_inflammation',
    'lips_irritation', 'itchiness', 'ulcers', 'toenail_loss',
    'speech_problem', 'bullseye_rash']

st.title("Yellow Fever Prediction")
st.info('Select symptoms and click Predict.')

# Collect user input
user_input = {}
for feature in features:
    user_input[feature] = 1 if st.checkbox(f"{feature.replace('_', ' ').capitalize()}") else 0

# Prediction block
if st.button("Predict Yellow Fever"):
    try:
        X = pd.DataFrame([user_input])

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        st.subheader("Prediction:")
        st.success("Yellow Fever Detected" if prediction == 1 else "No Yellow Fever Detected")

        st.subheader("Confidence:")
        st.write(f"{probability:.2%} chance of Yellow Fever")

        # Save prediction
        X['Prediction'] = prediction
        X['Probability'] = probability

        file_name = "predictions.xlsx"

        if os.path.exists(file_name):
            existing_df = pd.read_excel(file_name)
            updated_df = pd.concat([existing_df, X], ignore_index=True)
        else:
            updated_df = X

        updated_df.to_excel(file_name, index=False)
        st.success("Prediction saved successfully!")

        if st.checkbox("Show my selected symptoms"):
            st.dataframe(X.T)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


if os.path.exists("predictions.xlsx"):
    if st.checkbox("Show saved predictions"):
        saved_df = pd.read_excel("predictions.xlsx")
        st.subheader("All Saved Predictions")
        st.dataframe(saved_df)
