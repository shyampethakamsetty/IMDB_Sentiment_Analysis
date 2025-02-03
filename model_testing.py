import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load('p_model.pkl')  # Replace with your model file path
vectorizer = joblib.load('pvectorizer.pkl')  # Replace with your vectorizer file path

# Define the prediction function
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])  # Vectorize the input text
    prediction = model.predict(text_vectorized)[0]  # Predict the sentiment (0 or 1)
    return "Positive" if prediction == 1 else "Negative"

# Streamlit app
def main():
    st.title("Movie Review Sentiment Analysis")
    st.write("Upload a CSV file containing movie reviews, and get the sentiment predictions.")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)

        # Check if the required column 'review' exists
        if 'review' not in data.columns:
            st.error("The uploaded CSV file must contain a 'review' column.")
            return

        # Display the uploaded data
        st.write("Uploaded CSV Data:")
        st.dataframe(data.head())

        # Predict sentiment for each review
        data['sentiment'] = data['review'].apply(predict_sentiment)

        # Display results with predictions
        st.write("Sentiment Predictions:")
        st.dataframe(data[['review', 'sentiment']])

        # Allow users to download the results
        output_csv = data[['review', 'sentiment']].to_csv(index=False)
        st.download_button("Download Sentiment Predictions", output_csv, file_name="sentiment_predictions.csv", mime="text/csv")

if __name__ == "__main__":
    main()
