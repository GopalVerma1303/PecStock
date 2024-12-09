import streamlit as st
from utils import get_answer_csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.header("Forecast with AI ✨")

# Check if GROQ_API_KEY is set
if "GROQ_API_KEY" not in st.secrets:
    st.error("Please set the GROQ_API_KEY in your Streamlit secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    query = st.text_area("Ask any question related to the document")
    button = st.button("Submit")
    if button:
        with st.spinner("Processing your query..."):
            try:
                logger.info(f"Processing query: {query}")
                answer = get_answer_csv(uploaded_file, query)
                logger.info(f"Received answer: {answer}")

                # Display the answer
                st.subheader("Answer:")
                st.write(answer)

            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
