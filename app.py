%%writefile app.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io  # For capturing df.info()
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Global Finance Prediction", layout="wide")

# Load trained model
@st.cache_resource
def load_model():
    file_path = os.path.join(os.path.dirname(__file__), "gs_random_forest.pkl")
    return joblib.load(file_path)  

model = load_model()

# App title
st.title("Global Finance Multi-Class Prediction System")
st.write(
    """
    This application predicts financial indicators based on given features. 
    Upload a CSV file to get predictions. The model has been fine-tuned using RandomForestClassifier.
    Accuracy is set at 95%. Data was gotten from The Global Financial Development Database
    https://www.worldbank.org/en/publication/gfdr/data/global-financial-development-database
    the dataset in the link contains data dictionary of the features.
    Email me at nosakhareasowata94@gmail.com for feedback/remarks 
    as i will be updating my code from the feedbacks i get, thanks in anticipation😊.
    """
)

# Provide test file download link
st.write("### Download Test File (CSV Format) and make your predictions")
csv_url = "https://docs.google.com/spreadsheets/d/1AW6yl0CrERXPYng8Nzfih5vL6FNlYxoJJd6OleLnihE/export?format=csv"
st.markdown(f"[Download Sample Test File]({csv_url})")

st.write("### Download Test File results (CSV Format) and compare with model predictions")
csv_url = "https://docs.google.com/spreadsheets/d/1ZHtRcQeDEGguMpQMoPyyFR-2QwD1ZrnXIXXxgqPoaWI/export?format=csv"
st.markdown(f"[Download Sample Test File]({csv_url})") 


# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())
    
    # Show dataset information
    st.write("### Data Information:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    # Ensure the dataset has the expected features
    expected_features = model.feature_names_in_
    if all(feature in df.columns for feature in expected_features):
        predictions = model.predict(df[expected_features])
        df["Predicted_Label"] = predictions
        df["Income_Category"] = df["Predicted_Label"].apply(lambda x: "High Income" if x == 0 else "Low Income" if x == 1 else "Lower Middle Income" if x == 2 else "Upper Middle Income")
        
        st.write("### Predictions:")
        st.dataframe(df[["Predicted_Label", "Income_Category"].head()])
        
        # Visualization
        st.write("### Predictions Distribution:")
        plt.figure(figsize=(8, 5))
        sns.countplot(x="Income_Category", data=df)
        plt.title("Distribution of Predictions")
        st.pyplot(plt)
        
        # Download predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.error("Uploaded CSV does not match required features for prediction.")
