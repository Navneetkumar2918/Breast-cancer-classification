import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime


# ========================
# Page Config
# ========================
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")


# ========================
# Custom Button Styles (Responsive + Centered)
# ========================
st.markdown("""
<style>

/* Default Button Style */
div.stButton > button[kind="primary"],
div.stDownloadButton > button {

    background: linear-gradient(to right, #2193b0, #6dd5ed);
    color: white;

    border-radius: 12px;

    height: 50px;
    width: 260px;

    font-size: 18px;
    font-weight: bold;

    border: none;
}


/* Hover Effect */
div.stButton > button:hover,
div.stDownloadButton > button:hover {

    opacity: 0.85;
    transform: scale(1.03);
}


/* When Sidebar is Open / Small Screen */
@media (max-width: 900px) {

    div.stButton > button[kind="primary"],
    div.stDownloadButton > button {

        width: 200px;
        height: 42px;

        font-size: 15px;
    }
}


/* Force Center Alignment */
div.stButton,
div.stDownloadButton {

    display: flex;
    justify-content: center;
}

</style>
""", unsafe_allow_html=True)


# ========================
# Load Dataset
# ========================
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(
    breast_cancer_dataset.data,
    columns=breast_cancer_dataset.feature_names
)

data_frame['label'] = breast_cancer_dataset.target


# ========================
# Train Model
# ========================
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


# ========================
# Accuracy
# ========================
X_test_prediction = model.predict(X_test)
accuracy = accuracy_score(Y_test, X_test_prediction)


# ========================
# PDF Generator
# ========================
def generate_pdf(name, result, accuracy):

    file_name = "Medical_Report.pdf"

    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-70, "Breast Cancer Diagnosis Report")

    c.line(50, height-90, width-50, height-90)

    y = height-140

    c.setFont("Helvetica", 14)

    c.drawString(80, y, f"Patient Name: {name}")
    y -= 35

    c.drawString(80, y, f"Diagnosis Result: {result}")
    y -= 35

    c.drawString(80, y, f"Model Accuracy: {accuracy*100:.2f}%")
    y -= 35

    date = datetime.now().strftime("%d-%m-%Y  %H:%M")
    c.drawString(80, y, f"Date & Time: {date}")
    y -= 35

    c.drawString(80, y, "System: AI Based Medical Prediction System")

    # Footer
    c.line(50, 120, width-50, 120)

    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, 90, "Developed by Navneet Kumar")

    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, 70, "Machine Learning | AI in Healthcare")

    c.save()

    return file_name


# ========================
# Title
# ========================
st.markdown("<h1 style='text-align:center;'>🩺 Breast Cancer Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI Based Tumor Diagnosis System</p>", unsafe_allow_html=True)


# ========================
# Sidebar Info (Professional)
# ========================

st.sidebar.title("About This App")

st.sidebar.markdown("""
### 🩺 Breast Cancer Prediction App

This application helps in early detection of breast cancer by predicting
whether a tumor is **Malignant** or **Benign** using Machine Learning.

---

### ⚡ Features
- User-friendly interface  
- Real-time prediction  
- Medical PDF report  
- High accuracy model  
- Transparent results  

---

### 👨‍💻 Developer Info

<b>Navneet Kumar</b><br>
📧 Email: navneetkumar18112002@gmail.com<br>
🎓 B.Tech in Computer Science and Engineering<br>
🏫 National Institute of Technology Silchar

""", unsafe_allow_html=True)



# ========================
# Patient Name
# ========================
st.markdown("<h3 style='text-align:center;'>👤 Patient Information</h3>", unsafe_allow_html=True)

n1, n2, n3 = st.columns([1,2,1])

with n2:
    patient_name = st.text_input("Enter Patient Name")


# ========================
# Input Section
# ========================
st.markdown("<h3 style='text-align:center;'>📝 Enter Patient Data</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

input_data = []

for i, feature in enumerate(breast_cancer_dataset.feature_names):

    if i % 3 == 0:
        val = col1.number_input(feature, format="%.5f")

    elif i % 3 == 1:
        val = col2.number_input(feature, format="%.5f")

    else:
        val = col3.number_input(feature, format="%.5f")

    input_data.append(val)


st.markdown("<br>", unsafe_allow_html=True)


# ========================
# Predict Button (Centered)
# ========================
p1, p2, p3 = st.columns([1,2,1])

with p2:
    predict_clicked = st.button("🔍 Predict", type="primary")


# ========================
# Prediction
# ========================
if predict_clicked:

    if patient_name.strip() == "":
        st.warning("⚠️ Please enter patient name first.")

    else:

        input_array = np.asarray(input_data).reshape(1, -1)

        prediction = model.predict(input_array)

        st.markdown("<hr>", unsafe_allow_html=True)

        if prediction[0] == 0:

            result_text = "Malignant ❌"

            st.markdown(
                "<h3 style='text-align:center;color:red;'>❌ Malignant Tumor Detected</h3>",
                unsafe_allow_html=True
            )

        else:

            result_text = "Benign ✅"

            st.markdown(
                "<h3 style='text-align:center;color:green;'>✅ Benign Tumor Detected</h3>",
                unsafe_allow_html=True
            )


        # Accuracy
        st.markdown(
            f"<h3 style='text-align:center;'>📊 Accuracy: {accuracy*100:.2f}%</h3>",
            unsafe_allow_html=True
        )


        # Generate PDF
        pdf_file = generate_pdf(patient_name, result_text, accuracy)

        st.markdown("<br>", unsafe_allow_html=True)


        # ========================
        # Download Button (Centered)
        # ========================
        d1, d2, d3 = st.columns([1,2,1])

        with d2:
            st.download_button(
                "📄 Download Medical Report (PDF)",
                data=open(pdf_file, "rb"),
                file_name=pdf_file,
                mime="application/pdf"
            )


# ========================
# Footer
# ========================
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align:center;'>👨‍💻 Developed by Navneet Kumar</h4>

<p style='text-align:center; color:gray;'>
📧 Email: <a href="mailto:navneetkumar18112002@gmail.com" style="color:#4CAF50; text-decoration:none;">
navneetkumar18112002@gmail.com
</a><br>

<p style='text-align:center; color:gray;'>            
🎓 B.Tech in Computer Science and Engineering<br>
🏫 National Institute of Technology Silchar
</p>
""", unsafe_allow_html=True)


st.markdown("<p style='text-align:center;color:gray;'>Machine Learning Developer | AI in Healthcare</p>", unsafe_allow_html=True)


