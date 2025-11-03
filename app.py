
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time

# ---------------------------------------------------------------
#  SETUP AND MODEL LOADING
# ---------------------------------------------------------------
st.set_page_config(page_title="HAR Model Demo", layout="centered")
st.title("🧍 Human Activity Recognition (HAR) — Smart Fitness Tracker")

try:
    lr_model = joblib.load("log_reg_model.pkl")
    svm_model = joblib.load("SVM_model.pkl")
    model_loaded = True
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    model_loaded = False


# ---------------------------------------------------------------
#  SECTION 1 — MODEL SELECTION AND SINGLE PREDICTION
# ---------------------------------------------------------------
st.header("🎯 Test Your Model with Real Sensor Data")

st.markdown("Select a model and an activity to test predictions using your simulation dataset.")

model_choice = st.sidebar.selectbox("Choose a model:", ("Logistic Regression", "SVM"))

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("trial.csv")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

df = load_dataset()

activities = [
    "LAYING",
    "SITTING",
    "STANDING",
    "WALKING",
    "WALKING_DOWNSTAIRS",
    "WALKING_UPSTAIRS"
]
activity = st.selectbox("Select an Activity", activities)

# Stick figure image links
images = {
    "LAYING": "https://thumb.ac-illust.com/1c/1ceaf46eb0caec111c5d9ae59a5dbf5b_t.jpeg",
    "SITTING": "https://www.shutterstock.com/image-vector/stick-figure-sitting-icon-man-260nw-2311274887.jpg",
    "STANDING": "https://www.shutterstock.com/image-vector/stick-figure-man-person-stand-260nw-2279865073.jpg",
    "WALKING": "https://static.vecteezy.com/system/resources/thumbnails/007/126/667/small/person-walking-icon-free-vector.jpg",
    "WALKING_DOWNSTAIRS": "https://www.shutterstock.com/image-vector/printable-design-label-sticker-black-260nw-2307519657.jpg",
    "WALKING_UPSTAIRS": "https://thumbs.dreamstime.com/b/black-white-cartoon-man-walking-up-stairs-to-upper-floor-black-white-cartoon-man-walking-up-stairs-to-191013831.jpg"
}
st.image(images[activity], caption=f"{activity} Pose", width=250)

if st.button("🔍 Generate & Predict"):
    if not model_loaded:
        st.error("❌ Models not loaded.")
    elif df is None:
        st.error("❌ Dataset not loaded.")
    else:
        filtered = df[df.iloc[:, -1].str.upper() == activity.upper()]
        if filtered.empty:
            st.error(f"No data found for activity '{activity}'.")
        else:
            sample = filtered.sample(n=1, random_state=random.randint(0, 9999))
            X = sample.iloc[:, :-1]
            y_actual = sample.iloc[:, -1].values[0]

            st.subheader("📊 Sample Sensor Data")
            st.dataframe(X.head(1))

            # Model prediction
            if model_choice == "Logistic Regression":
                pred = lr_model.predict(X)[0]
            else:
                pred = svm_model.predict(X)[0]

            label_map = {
                0: "LAYING",
                1: "SITTING",
                2: "STANDING",
                3: "WALKING",
                4: "WALKING_DOWNSTAIRS",
                5: "WALKING_UPSTAIRS"
            }
            pred_label = label_map.get(pred, str(pred))

            st.subheader("🎯 Prediction Result")
            st.write(f"**Predicted Activity:** {pred_label}")
            st.write(f"**Actual Activity:** {y_actual}")

            if pred_label.lower() == y_actual.lower():
                st.success("✅ Model predicted correctly!")
            else:
                st.warning("⚠️ Prediction differs from actual.")


# ---------------------------------------------------------------
#  SECTION 2 — CONTINUOUS FITNESS TRACKER (SIMULATION)
# ---------------------------------------------------------------
import time

st.markdown("---")
st.header("🏃 Live Step & Calorie Tracker Simulation")

# Normalize activity labels
if df is not None:
    df['Activity'] = df['Activity'].astype(str).str.upper().str.strip()

# Select duration and simulation speed
sim_duration = st.slider("⏱️ Simulation Duration (in seconds)", 5, 30, 10)
interval = st.slider("⚡ Update Interval (seconds)", 1, 4, 1)

if st.button("▶️ Start Simulation"):
    if df is None:
        st.error("❌ No dataset found.")
    else:
        steps = 0
        calories = 0.0
        placeholder = st.empty()
        progress = st.progress(0)

        # Run simulation for selected duration
        for i in range(sim_duration):
            # Randomly choose one activity
            row = df.sample(1)
            activity = row.iloc[:, -1].values[0]

            # Step and calorie logic
            if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
                step_increase = random.randint(1, 3)
                steps += step_increase
                calories += 0.04 * step_increase  # basic approx
            elif activity == "STANDING":
                calories += 0.01
            elif activity == "SITTING":
                calories += 0.005
            else:
                calories += 0.002  # laying or unknown

            # Display simulation info
            placeholder.markdown(f"""
            ### 🧍 Detected Activity: **{activity}**
            - 🪜 Steps Count: `{steps}`
            - 🔥 Calories Burned: `{calories:.2f}`
            - ⏳ Time Elapsed: `{i+1} sec / {sim_duration} sec`
            """)

            progress.progress((i + 1) / sim_duration)
            time.sleep(interval)

        st.success("✅ Simulation Completed!")
        st.balloons()
