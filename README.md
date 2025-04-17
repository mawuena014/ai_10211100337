# AI-Powered Streamlit Web Application
# By: Mawuena Komla Ackotia - 10211100337

## Overview
This Streamlit web application showcases real-world applications of Artificial Intelligence (AI) and Machine Learning (ML), providing users with interactive modules for:
- **Regression Analysis:** Prediction and model evaluation.
- **K-Means Clustering:** Elbow method, clustering analysis, and visualization.
- **Neural Network Classifier:** Password strength detection using TensorFlow/Keras.
- **LLM-Powered Q&A System:** Retrieval-Augmented Generation (RAG) using Gemini API.
- **Data Preprocessing:** Missing value handling, feature scaling, encoding, and train-test splitting.

## Technologies Used
- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning & Deep Learning:** Scikit-learn, TensorFlow/Keras
- **Data Processing:** Pandas, PDFPlumber
- **Large Language Model (LLM):** Google Generative AI (Gemini API), LangChain, FAISS, Sentence Transformers
- **Visualization:** Matplotlib, Seaborn

## Application Structure
The application consists of five main sections:
1. **Home Page:** Displays course details and student information.
2. **Regression Page:** Enables linear regression modeling with interactive configurations.
3. **Clustering Page:** Implements K-Means clustering with optimal cluster selection.
4. **Neural Network Page:** Focuses on classification tasks such as password strength prediction.
5. **LLM Q&A System:** Supports document search and semantic retrieval.

## Features
### Regression Analysis
- **Data Handling:** CSV upload with preview, missing value handling, encoding, and feature selection.
- **Model Training:** Linear regression with adjustable train-test split.
- **Performance Metrics:** MSE, R² Score, MAE, Pearson Correlation.
- **Visualization:** Regression plots with actual vs predicted values.
- **Real-Time Predictions:** Interactive input for custom predictions.
- **Dataset:** Housing.csv  

### K-Means Clustering
- **Data Handling:** CSV upload with encoding fallbacks for real-world messy data.
- **Feature Selection:** Numeric features only, with optional scaling.
- **Clustering Method:** Elbow method to determine optimal clusters.
- **Visualization:** 2D/3D scatter plots, centroid markers, cluster distribution bar chart.
- **Downloadable Results:** Clustered dataset export in CSV format.
- **Dataset:** Housing.csv  

### Neural Network Classifier
- **Feature Engineering:** Password-based features (length, unique chars, entropy).
- **Customizable Model:** Adjustable architecture, batch size, learning rate.
- **Live Training Visualization:** Epoch-wise progress with validation loss/accuracy plots.
- **Evaluation Metrics:** Precision, recall, F1-score.
- **Manual & Batch Predictions:** Real-time password strength classification.
- **Dataset:** training_passwords.csv 

### LLM-Powered Q&A System (RAG)
- **Document Ingestion:** PDF/CSV text extraction with semantic chunking.
- **Vector Embedding:** Sentence Transformers with FAISS-based retrieval.
- **Confidence Scoring:** Semantic similarity, FAISS distance, and coverage ratio.
- **Answer Generation:** Gemini AI-driven response formulation.
- **Source Attribution:** Expandable section for retrieved document snippets.
- **Dataset:** Acity_Student_Handbook_2024 - Copy.pdf

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/mawuena014/ai_10211100337.git
   cd ai_10211100337
