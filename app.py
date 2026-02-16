import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Hyderabad Rent Estimator",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Premium Styling (High Contrast + Bold)
# --------------------------------------------------
st.markdown("""
<style>

/* Hide default streamlit header/menu/footer */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Remove extra padding */
.block-container {
    padding-top: 2rem;
}

/* Balanced visible background */
.stApp {
    background-image:
        linear-gradient(rgba(220,235,255,0.75), rgba(220,235,255,0.75)),
        url("https://images.unsplash.com/photo-1600607687920-4e2a09cf159d");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #0f172a;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 20px;
    font-weight: 500;
    color: #1e293b;
    margin-bottom: 40px;
}

/* Make labels bold & bigger */
label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* Input fields styling */
input, select, textarea {
    background-color: rgba(255,255,255,0.95) !important;
    color: #0f172a !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Button styling */
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 52px;
    font-size: 18px;
    font-weight: 700;
    background-color: #1e40af;
    color: white;
    border: none;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Result styling */
.result-text {
    margin-top: 45px;
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    color: #0b3c91;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 60px;
    font-size: 14px;
    font-weight: 500;
    color: #1e293b;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model, encoders = pickle.load(open("house_rent_model.pkl", "rb"))

# --------------------------------------------------
# UI Content
# --------------------------------------------------
st.markdown('<div class="main-title">🏠 Hyderabad House Rent Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Based Monthly Rental Estimation</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 6, 2)
    washrooms = st.number_input("Washrooms", 1, 5, 2)
    area = st.number_input("Area (sqft)", 300, 5000, 1200)

with col2:
    furnishing = st.selectbox("Furnishing Type", encoders["Furnishing"].classes_)
    tennants = st.selectbox("Preferred Tennants", encoders["Tennants"].classes_)
    locality = st.selectbox("Locality", encoders["Locality"].classes_)

if st.button("✨ Predict Rent"):

    furnishing_enc = encoders["Furnishing"].transform([furnishing])[0]
    tennants_enc = encoders["Tennants"].transform([tennants])[0]
    locality_enc = encoders["Locality"].transform([locality])[0]

    input_data = np.array([[bedrooms, washrooms,
                            furnishing_enc,
                            tennants_enc,
                            area,
                            locality_enc]])

    prediction = model.predict(input_data)
    predicted_rent = int(prediction[0])

    st.markdown(
        f'<div class="result-text">Estimated Monthly Rent: ₹ {predicted_rent:,}</div>',
        unsafe_allow_html=True
    )

    st.success("Prediction Generated Successfully 🎉")

st.markdown('<div class="footer">Built with Machine Learning • Powered by Streamlit • Designed by Dilliswari</div>', unsafe_allow_html=True)
