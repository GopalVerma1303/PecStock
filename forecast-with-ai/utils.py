import pandas as pd
import streamlit as st
import groq
import logging
from io import StringIO
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and pandas types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


def analyze_data(df):
    """
    Perform comprehensive analysis on the entire dataset.
    """
    analysis = {}

    # Basic statistics for all numeric columns
    numeric_stats = df.describe().to_dict()
    # Convert numpy types to Python native types
    numeric_stats = {k: {k2: float(v2) if isinstance(v2, np.number) else v2
                         for k2, v2 in v.items()}
                     for k, v in numeric_stats.items()}
    analysis['numeric_stats'] = numeric_stats

    # Trend analysis
    trends = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate overall trend
        if len(df) > 1:
            first_value = float(df[column].iloc[0])
            last_value = float(df[column].iloc[-1])
            total_change = ((last_value - first_value) /
                            first_value) * 100 if first_value != 0 else 0

            # Calculate recent trend (last 25% of data)
            recent_data = df[column].iloc[-len(df)//4:]
            recent_change = ((recent_data.iloc[-1] - recent_data.iloc[0]) /
                             recent_data.iloc[0]) * 100 if recent_data.iloc[0] != 0 else 0

            trends[column] = {
                'total_change_percent': float(round(total_change, 2)),
                'recent_change_percent': float(round(recent_change, 2)),
                'first_value': float(round(first_value, 2)),
                'last_value': float(round(last_value, 2)),
                'mean_value': float(round(df[column].mean(), 2)),
                'std_dev': float(round(df[column].std(), 2))
            }

    analysis['trends'] = trends

    # Seasonality analysis (if we have enough data)
    seasonality = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        if len(df) >= 12:
            # Calculate rolling statistics
            rolling_mean = df[column].rolling(window=12).mean()
            rolling_std = df[column].rolling(window=12).std()

            # Calculate peak and trough values
            peak_value = float(df[column].max())
            trough_value = float(df[column].min())
            peak_index = str(df[column].idxmax())
            trough_index = str(df[column].idxmin())

            seasonality[column] = {
                'peak_value': float(round(peak_value, 2)),
                'trough_value': float(round(trough_value, 2)),
                'peak_index': peak_index,
                'trough_index': trough_index,
                'volatility': float(round(rolling_std.mean(), 2)),
                'trend_strength': float(round(abs(rolling_mean.iloc[-1] - rolling_mean.iloc[0]), 2))
            }

    analysis['seasonality'] = seasonality

    # Correlation analysis
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        # Convert to a more readable format
        correlations = {}
        for col1 in corr_matrix.columns:
            correlations[col1] = {}
            for col2 in corr_matrix.columns:
                if col1 != col2:
                    correlations[col1][col2] = float(round(
                        corr_matrix.loc[col1, col2], 3))
        analysis['correlations'] = correlations

    # Data quality metrics
    quality_metrics = {}
    for column in df.columns:
        quality_metrics[column] = {
            'missing_values': int(df[column].isnull().sum()),
            'unique_values': int(df[column].nunique()),
            'data_type': str(df[column].dtype)
        }
    analysis['data_quality'] = quality_metrics

    return analysis


def generate_forecast(df, column_name, periods=5):
    """
    Generate forecast values using multiple methods and return the best one.
    """
    try:
        # Prepare data
        data = df[column_name].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Try different forecasting methods
        forecasts = {}

        # 1. Simple Moving Average
        ma_forecast = df[column_name].rolling(window=3).mean().iloc[-periods:]
        forecasts['Moving Average'] = ma_forecast

        # 2. Exponential Smoothing
        model = ExponentialSmoothing(df[column_name], seasonal_periods=12)
        fitted_model = model.fit()
        es_forecast = fitted_model.forecast(periods)
        forecasts['Exponential Smoothing'] = es_forecast

        # 3. ARIMA
        model = ARIMA(df[column_name], order=(1, 1, 1))
        fitted_model = model.fit()
        arima_forecast = fitted_model.forecast(steps=periods)
        forecasts['ARIMA'] = arima_forecast

        # Select the best forecast based on historical accuracy
        best_method = None
        best_accuracy = float('inf')

        for method, forecast in forecasts.items():
            if len(forecast) > 0:
                accuracy = float(np.mean(
                    np.abs(forecast - df[column_name].iloc[-len(forecast):])))
                if accuracy < best_accuracy:
                    best_accuracy = accuracy
                    best_method = method

        return forecasts[best_method], best_method, best_accuracy

    except Exception as e:
        logger.error(f"Error in forecast generation: {str(e)}")
        return None, None, None


def get_answer_csv(file, query: str) -> str:
    """
    Returns the answer to the given query by analyzing a CSV file using Groq API.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Perform comprehensive analysis
        analysis = analyze_data(df)

        # Generate forecasts for numeric columns
        forecast_results = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            forecast_values, method, accuracy = generate_forecast(df, column)
            if forecast_values is not None:
                forecast_results[column] = {
                    'values': [float(x) for x in forecast_values.tolist()],
                    'method': method,
                    'accuracy': float(round(accuracy, 2)) if accuracy is not None else None
                }

        # Create forecast table snippets
        forecast_tables = {}
        for column, result in forecast_results.items():
            current_data = df[column].tail(5).to_frame()
            forecast_data = pd.DataFrame({
                column: result['values']
            }, index=range(len(df), len(df) + len(result['values'])))

            forecast_tables[column] = {
                'current': current_data.to_string(),
                'forecast': forecast_data.to_string(),
                'method': result['method'],
                'accuracy': result['accuracy']
            }

        # Prepare comprehensive data summary
        data_summary = {
            'analysis': analysis,
            'forecasts': forecast_results,
            'current_data_sample': {k: [float(x) if isinstance(x, np.number) else x for x in v]
                                    for k, v in df.tail(5).to_dict().items()},
            'forecast_tables': forecast_tables,
            'total_rows': int(len(df)),
            'columns': list(df.columns)
        }

        # Convert to JSON for efficient transmission
        data_summary_json = json.dumps(
            data_summary, indent=2, cls=NumpyEncoder)

        # Prepare the prompt for Groq
        prompt = f"""Analyze the following comprehensive data analysis and answer the question:

        {data_summary_json}

        Question: {query}

        Provide a detailed answer based on the comprehensive data analysis, including specific forecasted values and their implications."""

        # Initialize Groq client
        client = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])

        # Make API call to Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert demand forecasting analyst. Your task is to provide active, data-driven insights based on the provided comprehensive data analysis. Follow these guidelines:

                    1. Data Analysis:
                       - Analyze the current data patterns using the provided statistics
                       - Identify key trends using the trend analysis
                       - Highlight seasonal patterns from the seasonality analysis
                       - Reference specific values from the correlation analysis
                       - Consider data quality metrics in your analysis

                    2. Forecasting Insights:
                       - Provide specific forecasted values from the data
                       - Explain the forecasting method used and its accuracy
                       - Compare current vs forecasted values
                       - Highlight significant changes or trends

                    3. Actionable Recommendations:
                       - Provide specific, data-backed recommendations
                       - Include exact numbers and percentages
                       - Suggest concrete actions based on the forecasts
                       - Highlight potential risks or opportunities

                    PROVIDE YOUR ANALYSIS IN THIS EXACT FORMAT:

                    CURRENT DATA ANALYSIS:
                    - Key Metrics: [list specific values from numeric_stats]
                    - Trends: [describe with exact numbers from trends]
                    - Seasonal Patterns: [identify with data points from seasonality]
                    - Correlations: [highlight significant relationships]
                    - Data Quality: [mention any important quality metrics]

                    FORECASTED VALUES:
                    - Method Used: [specify the forecasting method]
                    - Forecasted Values: [show actual numbers]
                    - Accuracy: [provide accuracy percentage]
                    - Comparison: [current vs forecasted with numbers]

                    RECOMMENDATIONS:
                    - Immediate Actions: [specific, data-backed steps]
                    - Expected Impact: [quantify with numbers]
                    - Risk Factors: [list with probabilities]
                    - Opportunities: [identify with potential values]

                    Base all recommendations on the actual data provided and include specific numbers from the forecasted values.""",
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
