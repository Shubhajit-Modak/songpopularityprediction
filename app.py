import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spotify Song Popularity Prediction",
    page_icon="🎵",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_package = joblib.load("models/best_model.pkl")
    return model_package["model"], model_package["features"]


model, features = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("🎵 Spotify Song Popularity Prediction")

st.write(
    "Predict a song's popularity category using nine Spotify audio features."
)

st.info(
    """
    **Popularity Categories**

    0–39 → Not Popular  
    40–69 → Moderately Popular  
    70–100 → Popular
    """
)


# ============================================================
# FEATURE INPUTS
# ============================================================

st.header("🎧 Enter Song Audio Features")

col1, col2 = st.columns(2)

with col1:

    danceability = st.slider(
        "Danceability",
        0.0,
        1.0,
        0.50,
        0.01,
        format="%.2f"
    )

    energy = st.slider(
        "Energy",
        0.0,
        1.0,
        0.50,
        0.01,
        format="%.2f"
    )

    speechiness = st.slider(
        "Speechiness",
        0.0,
        1.0,
        0.100,
        0.001,
        format="%.2f"
    )

    acousticness = st.slider(
        "Acousticness",
        0.0,
        1.0,
        0.500,
        0.001,
        format="%.2f"
    )

    instrumentalness = st.slider(
        "Instrumentalness",
        0.0,
        1.0,
        0.000,
        0.001,
        format="%.2f"
    )


with col2:

    loudness = st.number_input(
        "Loudness (dB)",
        min_value=-60.0,
        max_value=2.0,
        value=-10.0,
        step=0.1
    )

    liveness = st.slider(
        "Liveness",
        0.0,
        1.0,
        0.200,
        0.001
    )

    valence = st.slider(
        "Valence",
        0.0,
        1.0,
        0.500,
        0.001
    )

    tempo = st.number_input(
        "Tempo (BPM)",
        min_value=40.0,
        max_value=250.0,
        value=120.0,
        step=0.1
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Popularity",
    use_container_width=True
):

    new_song = pd.DataFrame(
        [[
            danceability,
            energy,
            loudness,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tempo
        ]],
        columns=features
    )

    prediction = model.predict(new_song)[0]

    probabilities = model.predict_proba(new_song)[0]

    probability_data = pd.DataFrame({
        "Category": model.classes_,
        "Probability": probabilities
    })

    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    st.header("📊 Prediction Result")

    if prediction == "Popular":
        st.success(
            f"🎉 Predicted Category: **{prediction}**"
        )

    elif prediction == "Moderately Popular":
        st.warning(
            f"🎧 Predicted Category: **{prediction}**"
        )

    else:
        st.error(
            f"📉 Predicted Category: **{prediction}**"
        )

    # --------------------------------------------------------
    # Probability chart
    # --------------------------------------------------------

    st.subheader("Prediction Probabilities")

    probability_data["Probability"] = (
        probability_data["Probability"] * 100
    )

    st.bar_chart(
        probability_data.set_index("Category")
    )

    # --------------------------------------------------------
    # Input values
    # --------------------------------------------------------

    st.subheader("Entered Audio Features")

    st.dataframe(
        new_song,
        use_container_width=True
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ About the Project")

st.write(
    """
    This project uses supervised machine learning to classify
    Spotify songs into three popularity categories.

    Three models were evaluated:
    
    • Logistic Regression  
    • Decision Tree  
    • Random Forest

    Random Forest was selected as the final model because it
    achieved the highest test performance.
    """
)

st.caption(
    "Final Random Forest accuracy: 74.43% | "
    "Weighted F1-score: 0.7456"
)