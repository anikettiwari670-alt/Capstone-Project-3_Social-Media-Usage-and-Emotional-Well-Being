import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Social Media Emotional Well-Being Predictor",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler.joblib"
COLUMNS_PATH = "model_columns.joblib"


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)

    return model, scaler, model_columns


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:

    model, scaler, model_columns = load_artifacts()

except Exception as e:

    st.error("Unable to load the model artifacts.")
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Social Media Usage & Emotional Well-Being")

st.write(
    "Predict the user's emotional state based on demographic information "
    "and social-media usage patterns."
)

st.divider()


# ============================================================
# USER INPUTS
# ============================================================

st.subheader("Enter User Information")


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    platform = st.selectbox(
        "Platform",
        [
            "Instagram",
            "Facebook",
            "Twitter",
            "YouTube",
            "TikTok",
            "LinkedIn",
            "Snapchat"
        ]
    )

    daily_usage = st.number_input(
        "Daily Usage Time (minutes)",
        min_value=0.0,
        value=120.0,
        step=1.0
    )


with col2:

    posts_per_day = st.number_input(
        "Posts Per Day",
        min_value=0.0,
        value=2.0,
        step=1.0
    )

    likes_per_day = st.number_input(
        "Likes Received Per Day",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    comments_per_day = st.number_input(
        "Comments Received Per Day",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    messages_per_day = st.number_input(
        "Messages Sent Per Day",
        min_value=0.0,
        value=20.0,
        step=1.0
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Emotional State",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Gender": [gender],

        "Platform": [platform],

        "Age": [age],

        "Daily_Usage_Time (minutes)": [daily_usage],

        "Posts_Per_Day": [posts_per_day],

        "Likes_Received_Per_Day": [likes_per_day],

        "Comments_Received_Per_Day": [comments_per_day],

        "Messages_Sent_Per_Day": [messages_per_day]
    })


    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        columns=["Gender", "Platform"]
    )


    # --------------------------------------------------------
    # ALIGN FEATURES WITH TRAINING COLUMNS
    # --------------------------------------------------------

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # SCALE FEATURES
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_scaled)

    predicted_class = prediction[0]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.success(
        f"Predicted Emotional State: **{predicted_class}**"
    )


    # --------------------------------------------------------
    # PREDICTION PROBABILITIES
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_scaled)[0]

        classes = model.classes_

        probability_df = pd.DataFrame({

            "Emotional State": classes,

            "Probability": probabilities

        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)


        st.subheader("Prediction Probabilities")

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )
