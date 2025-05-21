# 2023 Presidential Elections Sentiment Analysis

This project is a Streamlit web application and data science pipeline to analyze public sentiment around the 2023 Nigerian Presidential Elections using Twitter data. The application uses machine learning and natural language processing to classify the sentiment of tweets as positive, negative, or neutral, and provides interactive visualizations.

## Features

- **Real-time sentiment analysis** of user-input text.
- **Visualization** of sentiment distribution across tweets about key presidential candidates.
- **Machine learning pipeline** for text processing and sentiment classification.
- **Interactive web interface** using Streamlit.

## Project Structure

- `analysis.ipynb`: Jupyter notebook containing data loading, preprocessing, modeling, and exploratory analysis.
- `my_test.csv` (not included): The dataset of tweets used for analysis.
- `app.py` (if exists): Streamlit application for user interaction.
- `models/`: Directory to save trained machine learning models (if applicable).
- `requirements.txt`: List of Python dependencies.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aiyenigbacharles/2023-Presidential-Elections-Sentiment-Analysis.git
   cd 2023-Presidential-Elections-Sentiment-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data:**
   The first time you run the notebook/app, NLTK will attempt to download required resources.

### Running the Notebook

1. Open `analysis.ipynb` in Jupyter Notebook.
2. Run each cell in order to perform data cleaning, sentiment analysis, and model evaluation.

### Running the Streamlit App

If you have a `app.py` file:
```bash
streamlit run app.py
```

## Data

- The dataset (`my_test.csv`) should contain columns: `Date`, `User`, `Tweet`, `User_Location`.
- Tweets are preprocessed, labeled, and analyzed for sentiment using NLTK’s VADER and machine learning methods.

## Example Analysis

The notebook demonstrates:
- Data cleaning and tokenization
- Stopword removal and lemmatization
- Candidate extraction and labeling
- Sentiment scoring with VADER
- Model training and evaluation

## Technologies Used

- Python, pandas, matplotlib, scikit-learn, nltk, textblob, wordcloud, Streamlit

## License

Charles License

## Author

Charles Aiyenigba
