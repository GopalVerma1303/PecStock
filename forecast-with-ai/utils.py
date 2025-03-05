import pandas as pd
import streamlit as st
import groq
import logging
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_answer_csv(file, query: str) -> str:
    """
    Returns the answer to the given query by analyzing a CSV file using Groq API.

    Args:
    - file: The uploaded CSV file.
    - query (str): The question to ask about the CSV data.

    Returns:
    - answer (str): The answer to the query from the CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Get basic information about the CSV
        csv_info = f"CSV Info:\nColumns: {', '.join(df.columns)}\nShape: {df.shape}\n\nSample data:\n{df.head().to_string()}"

        # Prepare the prompt for Groq
        prompt = f"""Analyze the following CSV data and answer the question:

        {csv_info}

        Question: {query}

        Provide a detailed answer based on the CSV data."""

        # Initialize Groq client
        client = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])

        # Make API call to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert demand forecasting analyst. Your task is to thoroughly analyze the provided CSV data considering these key factors:

                    1. Data Pattern Analysis:
                       - Check for seasonality patterns
                       - Identify trends
                       - Look for cyclical patterns
                       - Assess data frequency (daily, monthly, yearly)
                       - Evaluate data completeness and quality
                       - Check for outliers and anomalies

                    2. Based on your data analysis, recommend the most suitable models from:

                    Quantitative Techniques:
                    - Historical Data Method (best for stable, consistent historical patterns)
                    - Time Series Analysis (for data with clear seasonal patterns)
                    - Econometric Modeling (for data with multiple influential variables)
                    - Predictive Sales Analytics (for complex patterns with multiple factors)
                    - Moving Averages (for smoothing short-term fluctuations)
                    - Regression Analysis (for clear linear relationships)
                    - ARIMA (for non-seasonal time series with trends)

                    Qualitative Techniques:
                    - Delphi Method (for long-term forecasting with expert opinions)
                    - Market Research (for new products/markets)
                    - Consumer Surveys (for direct consumer feedback)
                    - CPFR (for supply chain collaboration)

                    PROVIDE YOUR ANALYSIS IN THIS EXACT FORMAT:

                    DATA CHARACTERISTICS:
                    - Time Period: [specify the data timeframe]
                    - Pattern Type: [seasonal/trending/cyclical/random]
                    - Data Quality: [comment on completeness and reliability]
                    - Key Features: [list notable patterns or characteristics]

                    QUANTITATIVE ANALYSIS:
                    Best Model: [select based on data characteristics]
                    Expected Accuracy: [percentage based on data quality]
                    R² Score: [projected score based on data patterns]
                    Reason for Selection: [explain why this model fits the specific data patterns]
                    Implementation Guide: [data-specific implementation steps]

                    QUALITATIVE ANALYSIS:
                    Best Method: [select based on data gaps and business context]
                    Expected Accuracy: [percentage]
                    Reason for Selection: [explain why this method complements the quantitative approach]
                    Implementation Guide: [specific steps considering the data context]

                    Base all recommendations strictly on the analyzed CSV data patterns and characteristics. Do not provide generic responses.""",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=1000,
        )

        # Extract the response
        answer = chat_completion.choices[0].message.content
        logger.info(f"Received answer from Groq: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Error while processing query: {str(e)}")
        return f"An error occurred: {str(e)}"
