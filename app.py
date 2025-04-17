# --- Import necessary libraries ---
import streamlit as st  # Web app framework for creating interactive applications
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical operations and handling arrays
import matplotlib.pyplot as plt  # Visualization library for creating static plots
import seaborn as sns  # Statistical data visualization built on Matplotlib
import tensorflow as tf  # Machine learning framework for building deep learning models
import math  # Mathematical functions for numerical computations
import google.generativeai as genai  # Library for interfacing with Google's generative AI models
import pdfplumber  # Library for extracting text from PDFs
import time  # Module for managing time-related operations
from datetime import datetime  # Module for working with dates and timestamps
from sklearn.model_selection import train_test_split  # Function for splitting datasets into training and testing sets
from sklearn.linear_model import LinearRegression  # Linear regression model for predictive analysis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Performance metrics for regression models
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Tools for encoding categorical variables and scaling numerical data
from sklearn.cluster import KMeans  # Clustering algorithm for pattern recognition
from tensorflow.keras.models import Sequential  # Sequential model for deep learning architectures
from tensorflow.keras.layers import Dense  # Dense (fully connected) layers for neural networks
from scipy.stats import pearsonr  # Pearson correlation function for statistical analysis
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Library for text chunking and processing
from sentence_transformers import SentenceTransformer  # Pre-trained model for encoding sentences into embeddings
import faiss  # Library for efficient similarity search and clustering of dense vectors
from tenacity import retry, stop_after_attempt, wait_exponential  # Retry mechanisms for handling execution failures



# --- Function for Data Preprocessing ---
def preprocess_data(df, target_column):
    """Handles automatic categorical feature encoding and numerical scaling."""
    df = df.copy()
    label_encoders = {}

    # Encode ALL categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for later use (predictions)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical target variable if necessary
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    # Identify numeric features for scaling
    numerical_features = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    if len(numerical_features) > 0:  # Only scale if there are numerical columns
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X, y, scaler, label_encoders, categorical_features



# Navigation Bar Options Title
st.sidebar.title("Page Navigation Options")



# Sidebar navigation
page = st.sidebar.radio("Select a section", ["Home Page","Regression", "Clustering", "Neural Network", "Large Language Model (LLM)"])



# -----------------------------------------------
# Home Page
# -----------------------------------------------
if page == "Home Page":
    st.title("Introduction to Artificial Intelligence Project-Based Semester Examination")
    st.subheader("Name: Mawuena Komla Ackotia")
    st.subheader("Roll Number: 10211100337")
    st.subheader("Program: Computer Engineering")



# -----------------------------------------------
# Regression Section
# -----------------------------------------------
elif page == "Regression":
    st.header("Regression Analysis")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        
        # 1. Dataset Preview
        st.subheader("Dataset Preview")
        st.write(df.head(10))
        
        # 2. Data Preprocessing Options
        st.subheader("Data Preprocessing Options")
        
        # Missing value handling
        missing_values_option = st.selectbox(
            "Handle missing values:",
            ["Keep as-is", "Drop rows with missing values", "Impute with median"],
            index=0
        )
        
        # Feature scaling toggle
        enable_scaling = st.checkbox(" Enable feature scaling", value=True)
        
        # Apply preprocessing
        processed_df = df.copy()
        
        # Handle missing values
        if missing_values_option == "Drop rows with missing values":
            processed_df = processed_df.dropna()
            st.info(f"Dropped rows with missing values. New shape: {processed_df.shape}")
        elif missing_values_option == "Impute with median":
            numeric_cols = processed_df.select_dtypes(include=np.number).columns
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
            st.info("Imputed missing values with median for numeric columns")

        # Target column selection - only allow 'price' and 'area'
        allowed_targets = ['price', 'area']
        available_targets = [col for col in processed_df.columns if col in allowed_targets]
        
        if not available_targets:
            st.error("No valid target columns found (requires 'price' or 'area').")
            st.stop()
            
        target_column = st.selectbox(
            "Select the target column:", 
            available_targets
        )
        
        # Feature selection
        st.subheader("Feature Selection")
        feature_columns = [col for col in processed_df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select features for regression", 
            feature_columns,
            default=feature_columns
        )

        if not selected_features:
            st.error("Please select at least one feature for the model.")
            st.stop()

        # Create a unique key for the current configuration
        current_config = {
            'target': target_column,
            'features': sorted(selected_features),
            'missing': missing_values_option,
            'scaling': enable_scaling,
            'data_hash': hash(processed_df.to_csv())
        }
        config_hash = hash(str(current_config))

        # Clear previous results if configuration changed
        if 'regression_config_hash' in st.session_state and st.session_state['regression_config_hash'] != config_hash:
            for key in ['regression_model', 'regression_metrics', 'regression_plots']:
                if key in st.session_state:
                    del st.session_state[key]

        # Always rerun regression (to ensure fresh results)
        try:
            with st.spinner('Running regression analysis...'):
                # Clear previous plots
                if 'regression_plots' in st.session_state:
                    del st.session_state['regression_plots']
                
                # Preprocess data
                X = processed_df[selected_features]
                y = processed_df[target_column]
                
                # Encode categorical features
                label_encoders = {}
                categorical_features = X.select_dtypes(include=['object']).columns
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le
                
                # Scale features if enabled
                scaler = StandardScaler() if enable_scaling else None
                if enable_scaling:
                    numerical_features = X.select_dtypes(include=np.number).columns
                    X[numerical_features] = scaler.fit_transform(X[numerical_features])
                
                # Store preprocessing objects
                st.session_state.update({
                    'regression_scaler': scaler,
                    'regression_label_encoders': label_encoders,
                    'regression_categorical_features': categorical_features,
                    'regression_selected_features': selected_features,
                    'regression_config_hash': config_hash
                })

                # Train-test split with sample count display
                st.subheader("Train-Test Split")
                test_size = st.slider("Select test data percentage", 10, 50, 20)
                
                # Display sample counts immediately after slider
                total_samples = len(X)
                test_samples = int(total_samples * test_size/100)
                train_samples = total_samples - test_samples
                st.write(f"Training samples: {train_samples} ({100-test_size}%)")
                st.write(f"Test samples: {test_samples} ({test_size}%)")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size/100, 
                    random_state=42
                )

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                st.session_state['regression_model'] = model

                # Evaluate model
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_corr = np.corrcoef(y_train, y_train_pred)[0, 1]
                
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_corr = np.corrcoef(y_test, y_test_pred)[0, 1]

                # Display metrics
                st.subheader("Model Performance")
                
                st.write("**Training Data:**")
                st.write(f"- Mean Squared Error (MSE): {train_mse:,.2f}")
                st.write(f"- R² Score: {train_r2:.4f}")
                st.write(f"- Mean Absolute Error (MAE): {train_mae:,.2f}")
                st.write(f"- Correlation: {train_corr:.4f}")
                
                st.write("\n**Test Data:**")
                st.write(f"- Mean Squared Error (MSE): {test_mse:,.2f}")
                st.write(f"- R² Score: {test_r2:.4f}")
                st.write(f"- Mean Absolute Error (MAE): {test_mae:,.2f}")
                st.write(f"- Correlation: {test_corr:.4f}")

                # 3. Visualization of Regression Line
                st.subheader("3. Visualization of Regression Line")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Training set plot with red line of best fit
                sns.regplot(x=y_train, y=y_train_pred, ax=ax1, line_kws={'color': 'red'})
                ax1.set_title("Training Set: Actual vs Predicted")
                ax1.set_xlabel("Actual Values")
                ax1.set_ylabel("Predicted Values")
                ax1.ticklabel_format(style='plain')
                ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
                
                # Test set plot with red line of best fit
                sns.regplot(x=y_test, y=y_test_pred, ax=ax2, line_kws={'color': 'red'})
                ax2.set_title("Test Set: Actual vs Predicted")
                ax2.set_xlabel("Actual Values")
                ax2.set_ylabel("Predicted Values")
                ax2.ticklabel_format(style='plain')
                ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
                
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in regression analysis: {str(e)}")
            st.stop()

        # Prediction section
        if 'regression_model' in st.session_state:
            st.subheader("Make Predictions")
            
            # Get stored objects
            model = st.session_state['regression_model']
            scaler = st.session_state['regression_scaler']
            label_encoders = st.session_state.get('regression_label_encoders', {})
            categorical_features = st.session_state.get('regression_categorical_features', [])
            selected_features = st.session_state['regression_selected_features']

            # Create input form
            input_values = {}
            with st.form("prediction_form"):
                for feature in selected_features:
                    if feature == 'parking':
                        input_values[feature] = st.selectbox(
                            f"Select {feature}", 
                            options=[0, 1, 2, 3],
                            key=f"input_{feature}"
                        )
                    elif feature in ['bedrooms', 'bathrooms', 'stories']:
                        input_values[feature] = st.number_input(
                            f"Enter {feature} (integer)", 
                            value=int(processed_df[feature].median()),
                            step=1,
                            key=f"input_{feature}"
                        )
                    elif feature in categorical_features:
                        options = processed_df[feature].unique().tolist()
                        input_values[feature] = st.selectbox(
                            f"Select {feature}", 
                            options,
                            key=f"input_{feature}"
                        )
                    else:
                        input_values[feature] = st.number_input(
                            f"Enter {feature}", 
                            value=float(processed_df[feature].mean()),
                            key=f"input_{feature}"
                        )
                
                submitted = st.form_submit_button("Predict")
            
            if submitted:
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_values])
                    
                    # Convert specific features to integers
                    for feature in ['bedrooms', 'bathrooms', 'stories', 'parking']:
                        if feature in input_df.columns:
                            input_df[feature] = input_df[feature].astype(int)
                    
                    # Encode categorical features
                    for feature in categorical_features:
                        if feature in input_values:
                            input_df[feature] = label_encoders[feature].transform([input_values[feature]])
                    
                    # Scale features if enabled
                    if scaler is not None:
                        numerical_features = input_df.select_dtypes(include=np.number).columns
                        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
                    
                    # Make prediction
                    prediction = model.predict(input_df)
                    st.success(f"Predicted {target_column}: {prediction[0]:,.2f}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")



# -----------------------------------------------
# Clustering Section
# -----------------------------------------------
elif page == "Clustering":
    st.header("K-Means Clustering Analysis")
    
    # Import required for color mapping
    from matplotlib.colors import ListedColormap
    
    # 1. File Upload with Robust CSV Reading
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], key="cluster_upload")
    
    if uploaded_file is not None:
        try:
            # First attempt - try reading normally
            try:
                df = pd.read_csv(uploaded_file)
                st.success("CSV read successfully with default parameters")
            except Exception as e:
                st.warning(f"Default read failed: {str(e)}. Trying alternative methods...")
                
                # Second attempt - try different encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"Successfully read with {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    st.error("Failed to read with standard encodings. Trying engine='python'")
                    
                    # Third attempt - use python engine which is more forgiving
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, engine='python', error_bad_lines=False)
                    st.success("Read successfully with python engine")
            
            # Verify we got data
            if df.empty:
                st.error("The file appears to be empty or couldn't be parsed")
                st.stop()
                
            # Show basic info
            st.subheader("1. Dataset Preview")
            st.write(f"Shape: {df.shape} (rows × columns)")
            st.write("First 5 rows:")
            st.write(df.head())
            
            # 2. Data Cleaning
            st.subheader("2. Data Cleaning")
            
            # Show missing values
            if df.isnull().sum().sum() > 0:
                st.warning("Missing values detected:")
                st.write(df.isnull().sum())
                
                # Cleaning options
                clean_method = st.radio("Handle missing values:",
                                      ["Drop rows with missing values",
                                       "Fill with median (numeric) / mode (categorical)",
                                       "Fill with zeros"])
                
                if clean_method == "Drop rows with missing values":
                    df = df.dropna()
                elif clean_method == "Fill with median (numeric) / mode (categorical)":
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df = df.fillna(0)
                
                st.info(f"New shape after cleaning: {df.shape}")
            
            # 3. Feature Selection
            st.subheader("3. Feature Selection")
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if not numeric_cols:
                st.error("No numeric columns found for clustering. Please check your data.")
                st.stop()
                
            selected_features = st.multiselect(
                "Select numeric features for clustering:",
                numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
            )
            
            if len(selected_features) < 2:
                st.error("Please select at least 2 numeric features")
                st.stop()
            
            # 4. Clustering Setup
            st.subheader("4. Clustering Setup")
            
            # Scale features
            scale_data = st.checkbox("Scale features (recommended)", value=True)
            X = df[selected_features].copy()
            
            if scale_data:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            # Determine optimal clusters
            max_clusters = min(10, len(X_scaled)-1)
            
            with st.spinner("Calculating optimal cluster count..."):
                wcss = []
                for i in range(1, max_clusters+1):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)
            
            # Elbow plot with proper number formatting
            fig1, ax1 = plt.subplots(figsize=(8,4))
            ax1.plot(range(1, max_clusters+1), wcss, marker='o', linestyle='--')
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("WCSS")
            ax1.set_title("Elbow Method")
            
            # Format y-axis to show full numbers
            ax1.get_yaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, p: format(int(x), ','))
            )
            ax1.grid(True)
            st.pyplot(fig1)
            
            # Cluster selection
            k = st.slider("Select number of clusters:",
                         min_value=2,
                         max_value=max_clusters,
                         value=3 if max_clusters >=3 else 2)
            
            # 5. Run Clustering
            with st.spinner(f"Clustering into {k} groups..."):
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                df['Cluster'] = clusters
                
                # Get centroids in original scale
                if scale_data:
                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                else:
                    centroids = kmeans.cluster_centers_
            
            # 6. Visualization
            st.subheader("5. Cluster Visualization")
            
            # Configure number formatting for all plots
            plt.rcParams['axes.formatter.useoffset'] = False
            plt.rcParams['axes.formatter.use_mathtext'] = False
            
            # Get consistent color palette for all visualizations
            cluster_colors = sns.color_palette('viridis', k)
            
            # 2D Plot with proper formatting
            if len(selected_features) == 2:
                fig2, ax2 = plt.subplots(figsize=(10,6))
                
                # Scatter plot with clusters using consistent colors
                scatter = ax2.scatter(
                    X.iloc[:, 0], X.iloc[:, 1], 
                    c=clusters, cmap=ListedColormap(cluster_colors), 
                    alpha=0.6, s=50, edgecolor='w'
                )
                
                # Plot centroids
                ax2.scatter(
                    centroids[:, 0], centroids[:, 1],
                    marker='X', s=200, c='red',
                    label='Centroids'
                )
                
                # Axis labels with proper rotation
                ax2.set_xlabel(selected_features[0], rotation=0)
                ax2.set_ylabel(selected_features[1], rotation=90)
                
                # Format numbers on axes
                ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
                
                ax2.set_title(f"2D Cluster Visualization (k={k})")
                ax2.legend()
                ax2.grid(True)
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                st.pyplot(fig2)
            
            # 3D Plot if 3+ features selected
            elif len(selected_features) >= 3:
                fig3 = plt.figure(figsize=(10,8))
                ax3 = fig3.add_subplot(111, projection='3d')
                
                # Scatter plot with clusters using consistent colors
                scatter = ax3.scatter(
                    X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2],
                    c=clusters, cmap=ListedColormap(cluster_colors),
                    alpha=0.6, s=50, edgecolor='w'
                )
                
                # Centroids
                ax3.scatter(
                    centroids[:, 0], centroids[:, 1], centroids[:, 2],
                    marker='X', s=200, c='red',
                    label='Centroids'
                )
                
                # Axis labels with proper rotation
                ax3.set_xlabel(selected_features[0], rotation=0)
                ax3.set_ylabel(selected_features[1], rotation=0)
                ax3.set_zlabel(selected_features[2], rotation=0)
                
                # Format numbers on all axes
                ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
                ax3.zaxis.set_major_formatter(plt.FuncFormatter(lambda z, p: format(int(z), ',')))
                
                ax3.set_title(f"3D Cluster Visualization (k={k})")
                ax3.legend()
                
                # Adjust viewing angle for better visibility
                ax3.view_init(elev=20, azim=30)
                st.pyplot(fig3)
            
            # 7. Cluster Analysis
            st.subheader("6. Cluster Analysis")
            
            # Show cluster stats with proper number formatting
            st.write("Cluster Statistics (Mean Values):")
            cluster_stats = df.groupby('Cluster')[selected_features].mean()
            
            # Format all numbers in the dataframe
            styled_stats = cluster_stats.style.format("{:,.2f}")
            st.dataframe(styled_stats)
            
            # Cluster sizes with proper formatting
            st.write("Points per Cluster:")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            
            fig4, ax4 = plt.subplots(figsize=(8,4))
            bars = ax4.bar(
                cluster_counts.index, 
                cluster_counts.values,
                color=cluster_colors)  # Using the same color palette
            
            ax4.set_xlabel("Cluster", rotation=0)
            ax4.set_ylabel("Number of Points", rotation=90)
            ax4.set_title("Cluster Distribution")
            
            # Format y-axis numbers
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f"{int(height):,}",
                    ha='center', va='bottom'
                )
            
            st.pyplot(fig4)
            
            # 8. Download Results
            st.subheader("7. Download Results")
            st.download_button(
                "Download Clustered Data",
                df.to_csv(index=False),
                "clustered_data.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.error("Please check your CSV file format and try again")
            st.stop()



# -----------------------------------------------
# Neural Network Section
# -----------------------------------------------
elif page == "Neural Network":
    st.header("Neural Network Classifier")
    
    # Initialize session state variables if they don't exist
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'test_file_processed' not in st.session_state:
        st.session_state.test_file_processed = False

    # 1. Dataset Upload and Setup
    st.subheader("1. Dataset Upload")
    uploaded_file = st.file_uploader("Upload your training dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read dataset with encoding fallbacks
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    
                    # Clean data - remove rows with missing values
                    df = df.dropna()
                    if len(df) == 0:
                        raise ValueError("Dataset is empty after cleaning")
                        
                    break
                except Exception as e:
                    continue
            
            if df is None:
                st.error("Failed to read file. Please check the format.")
                st.stop()
                
            st.success(f"Dataset loaded successfully! {len(df)} rows remaining after cleaning.")
            st.write("Preview:", df.head())
            st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Enhanced Feature Engineering for Passwords
            if 'password' in df.columns:
                # Handle NaN passwords
                df['password'] = df['password'].fillna('')
                
                # Extract detailed password complexity features
                df['length'] = df['password'].str.len()
                df['upper_count'] = df['password'].str.count(r'[A-Z]')
                df['lower_count'] = df['password'].str.count(r'[a-z]')
                df['digit_count'] = df['password'].str.count(r'\d')
                df['special_count'] = df['password'].str.count(r'[^A-Za-z0-9]')
                df['unique_chars'] = df['password'].apply(lambda x: len(set(x)))
                df['entropy'] = df['password'].apply(
                    lambda x: -sum((x.count(c)/len(x))*math.log(x.count(c)/len(x)) for c in set(x)) 
                    if len(x) > 0 else 0
                )
                
                # Drop raw password column
                df = df.drop('password', axis=1)
                st.write("Enhanced Features Preview:", df.head())
            
            # 2. Target and Feature Selection
            st.subheader("2. Select Target and Features")
            
            # Show class distribution with proper labels
            if 'strength' in df.columns:
                fig, ax = plt.subplots()
                counts = df['strength'].value_counts().sort_index()
                bars = ax.bar(counts.index, counts.values)
                
                # Set proper x-axis labels
                ax.set_xticks(counts.index)
                ax.set_xticklabels(counts.index, rotation=0)  # Changed from 90 to 0 degrees
                ax.set_xlabel('Strength')
                ax.set_ylabel('Count')
                ax.set_title('Password Strength Distribution')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom')
                
                st.pyplot(fig)
            
            # Target column selection - only allow 'strength'
            if 'strength' not in df.columns:
                st.error("Required 'strength' column not found in dataset.")
                st.stop()
                
            target_column = 'strength'  # Force selection of strength column
            st.info(f"Target column set to: {target_column}")
            
            # Handle target encoding
            y = df[target_column]
            if df[target_column].dtype in ['object', 'category']:
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y.fillna('Missing'))
                original_classes = target_encoder.classes_
            else:
                y_encoded = y.fillna(0).astype(int)
                if min(y_encoded) != 0:
                    y_encoded = y_encoded - min(y_encoded)
                target_encoder = None
                original_classes = sorted(y.unique())
            
            # Feature selection
            feature_columns = [col for col in df.columns if col != target_column]
            selected_features = st.multiselect(
                "Select features:", 
                feature_columns,
                default=feature_columns
            )
            
            if len(selected_features) == 0:
                st.error("Please select at least one feature")
                st.stop()
            
            # 3. Data Preprocessing
            st.subheader("3. Data Preprocessing")
            X = df[selected_features].copy()
            
            # Encode categorical features
            categorical_features = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('Missing'))
                label_encoders[col] = le
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-test split
            test_size = st.slider("Test set size (%)", 10, 40, 20)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, 
                test_size=test_size/100, 
                random_state=42,
                stratify=y_encoded
            )
            
            # 4. Model Configuration
            st.subheader("4. Model Configuration")
            
            # Handle class imbalance
            from sklearn.utils import class_weight
            classes = np.unique(y_train)
            weights = class_weight.compute_class_weight(
                'balanced', 
                classes=classes, 
                y=y_train
            )
            class_weights = dict(zip(classes, weights))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.slider("Epochs", 2, 100, 5)
            with col2:
                batch_size = st.slider("Batch size", 8, 128, 8)
            with col3:
                learning_rate = st.slider("Learning rate", 0.0001, 0.1, 0.01, step=0.001)
            
            # Model architecture
            st.markdown("**Network Architecture**")
            hidden_layers = st.number_input("Hidden layers", 1, 5, 2)
            layer_units = []
            for i in range(hidden_layers):
                units = st.slider(f"Layer {i+1} units", 8, 256, 8)
                layer_units.append(units)
            
            # 5. Model Training
            st.subheader("5. Model Training")
            
            if st.button("Train Model"):
                tf.keras.backend.clear_session()
                model = Sequential()
                model.add(Dense(layer_units[0], activation='relu', input_shape=(X_train.shape[1],)))
                
                for units in layer_units[1:]:
                    model.add(Dense(units, activation='relu'))
                
                if len(original_classes) > 2:
                    model.add(Dense(len(original_classes), activation='softmax'))
                    loss_fn = 'sparse_categorical_crossentropy'
                else:
                    model.add(Dense(1, activation='sigmoid'))
                    loss_fn = 'binary_crossentropy'
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=loss_fn,
                    metrics=['accuracy']
                )
                
                # Training visualization setup
                history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
                progress_bar = st.progress(0)
                status_text = st.empty()
                plot_placeholder = st.empty()
                
                class Callback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        history['loss'].append(logs['loss'])
                        history['val_loss'].append(logs['val_loss'])
                        history['accuracy'].append(logs['accuracy'])
                        history['val_accuracy'].append(logs['val_accuracy'])
                        
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch {epoch+1}/{epochs}\n"
                                      f"Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}\n"
                                      f"Accuracy: {logs['accuracy']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        ax1.plot(history['loss'], label='Train')
                        ax1.plot(history['val_loss'], label='Validation')
                        ax1.set_title('Loss')
                        ax1.legend()
                        
                        ax2.plot(history['accuracy'], label='Train')
                        ax2.plot(history['val_accuracy'], label='Validation')
                        ax2.set_title('Accuracy')
                        ax2.legend()
                        
                        plot_placeholder.pyplot(fig)
                        plt.close()
                
                model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=class_weights,
                    callbacks=[Callback()],
                    verbose=0
                )
                
                # Store model and preprocessing
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['label_encoders'] = label_encoders
                st.session_state['target_encoder'] = target_encoder
                st.session_state['features'] = selected_features
                st.session_state['classes'] = original_classes
                st.session_state['model_trained'] = True
                
                st.success("Training complete!")
                
                # Show evaluation metrics
                y_pred = np.argmax(model.predict(X_test), axis=1) if len(original_classes) > 2 else (model.predict(X_test) > 0.5).astype(int)
                st.text(classification_report(y_test, y_pred, target_names=[str(c) for c in original_classes]))
        except:
            print("")
            

    # 6. Make Predictions (separate section that persists after training)
    if st.session_state.get('model_trained', False):
        st.subheader("6. Make Predictions")
        
        # Option 1: Upload test file
        st.markdown("**Option 1: Upload test data**")
        test_file = st.file_uploader("Upload test data (CSV)", type=["csv"], key="test_upload")
        
        if test_file is not None:
            try:
                test_df = pd.read_csv(test_file)
                
                # Feature engineering for test data
                if 'password' in test_df.columns:
                    test_df['password'] = test_df['password'].fillna('')
                    test_df['length'] = test_df['password'].str.len()
                    test_df['upper_count'] = test_df['password'].str.count(r'[A-Z]')
                    test_df['lower_count'] = test_df['password'].str.count(r'[a-z]')
                    test_df['digit_count'] = test_df['password'].str.count(r'\d')
                    test_df['special_count'] = test_df['password'].str.count(r'[^A-Za-z0-9]')
                    test_df['unique_chars'] = test_df['password'].apply(lambda x: len(set(x)))
                    test_df['entropy'] = test_df['password'].apply(
                        lambda x: -sum((x.count(c)/len(x))*math.log(x.count(c)/len(x)) for c in set(x)) 
                        if len(x) > 0 else 0
                    )
                
                # Select features
                X_test_new = test_df[st.session_state['features']].copy()
                
                # Encode categorical features
                for col in st.session_state['label_encoders']:
                    le = st.session_state['label_encoders'][col]
                    X_test_new[col] = le.transform(X_test_new[col].astype(str).fillna('Missing'))
                
                # Scale features
                X_test_scaled = st.session_state['scaler'].transform(X_test_new)
                
                # Make predictions
                model = st.session_state['model']
                predictions = model.predict(X_test_scaled)
                
                if len(st.session_state['classes']) > 2:
                    predicted_classes = np.argmax(predictions, axis=1)
                else:
                    predicted_classes = (predictions > 0.5).astype(int)
                
                if st.session_state['target_encoder']:
                    test_df['Prediction'] = st.session_state['target_encoder'].inverse_transform(predicted_classes)
                else:
                    test_df['Prediction'] = [st.session_state['classes'][i] for i in predicted_classes]
                
                st.write("Predictions:", test_df)
                
                # Download predictions
                csv = test_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )
                
            except Exception as e:
                st.error(f"Error processing test file: {str(e)}")
        
        # Option 2: Manual prediction
        st.markdown("**Option 2: Enter password manually**")
        password = st.text_input("Enter password:", type="password", key="manual_pred")
        
        if st.button("Predict Strength", key="predict_btn"):
            try:
                # Create enhanced features from password
                features = {
                    'length': len(password),
                    'upper_count': sum(1 for c in password if c.isupper()),
                    'lower_count': sum(1 for c in password if c.islower()),
                    'digit_count': sum(1 for c in password if c.isdigit()),
                    'special_count': sum(1 for c in password if not c.isalnum()),
                    'unique_chars': len(set(password)),
                    'entropy': -sum((password.count(c)/len(password))*math.log(password.count(c)/len(password)) 
                              for c in set(password)) if len(password) > 0 else 0
                }
                
                # Create input dataframe
                input_df = pd.DataFrame([features])
                input_df = input_df[st.session_state['features']]  # Ensure correct feature order
                
                # Scale features
                input_scaled = st.session_state['scaler'].transform(input_df)
                
                # Predict
                model = st.session_state['model']
                pred = model.predict(input_scaled)
                
                if len(st.session_state['classes']) > 2:
                    pred_class = np.argmax(pred)
                    proba = pred[0][pred_class]
                else:
                    pred_class = int(pred > 0.5)
                    proba = pred[0][0] if pred_class == 1 else 1 - pred[0][0]
                
                if st.session_state['target_encoder']:
                    pred_label = st.session_state['target_encoder'].inverse_transform([pred_class])[0]
                else:
                    pred_label = st.session_state['classes'][pred_class]
                
                st.success(f"Predicted Strength: {pred_label} (Confidence: {proba:.2%})")
                
                # Show feature breakdown
                st.subheader("Password Analysis")
                analysis_df = pd.DataFrame({
                    'Feature': ['Length', 'Uppercase', 'Lowercase', 'Digits', 'Special', 'Unique', 'Entropy'],
                    'Count': [
                        features['length'],
                        features['upper_count'],
                        features['lower_count'],
                        features['digit_count'],
                        features['special_count'],
                        features['unique_chars'],
                        features['entropy']
                    ]
                })
                st.table(analysis_df)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")



# -----------------------------------------------
# Large Language Model (LLM) Section
# -----------------------------------------------
elif page == "Large Language Model (LLM)":
    # --- Step 1: API Key with Model Selection ---
    st.header("🔐 API Configuration")
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    
    # API key input with help link
    col1, col2 = st.columns([3, 2])
    with col1:
        st.session_state.api_key = st.text_input("Paste your Gemini API Key:", type="password")
    with col2:
        st.markdown(
            f"""
            <div style="margin-top: 1.8rem;">
                <a href="https://ai.google.dev/gemini-api/docs/api-key?authuser=1" 
                   target="_blank" 
                   style="color: #1a73e8; text-decoration: none;
                          font-size: 0.9rem; display: inline-block;
                          padding: 0.5rem 1rem; border-radius: 4px;
                          border: 1px solid #dadce0;">
                    Don't have a Gemini API key? Get one
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Model selection
    model_choice = st.radio(
        "Model Priority:",
        ["Fastest (gemini-1.5-flash)", "Most Capable (gemini-1.5-pro)"],
        horizontal=True
    )

    if st.session_state.api_key:
        genai.configure(api_key=st.session_state.api_key)

        # Cache everything for speed
        @st.cache_resource
        def load_components():
            embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )
            return embedder, text_splitter

        embedder, text_splitter = load_components()

        # --- Safe Confidence Calculation ---
        def safe_cosine_similarity(vec1, vec2):
            """Calculate cosine similarity with array safety checks"""
            if vec1 is None or vec2 is None:
                return 0.0
            if vec1.ndim == 1:
                vec1 = vec1.reshape(1, -1)
            if vec2.ndim == 1:
                vec2 = vec2.reshape(1, -1)
            
            try:
                dot_product = np.dot(vec1, vec2.T)
                norm_product = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
                return np.clip(dot_product / norm_product, -1.0, 1.0)
            except Exception:
                return 0.0

        def calculate_context_confidence(query_embed, chunks, indices, distances):
            """Enhanced confidence calculation with complete bounds checking"""
            if not hasattr(indices, 'size') or indices.size == 0 or not chunks:
                return 0.3
            
            valid_indices = []
            similarities = []
            distance_scores = []
            
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(chunks):
                    try:
                        chunk_embed = embedder.encode(chunks[idx])
                        sim = safe_cosine_similarity(query_embed, chunk_embed)
                        similarities.append(sim[0][0] if hasattr(sim, 'ndim') and sim.ndim > 1 else sim)
                        
                        # Convert FAISS distance to similarity score (0-1)
                        if i < len(distances[0]):
                            dist_score = 1 - min(1.0, distances[0][i] / 2)  # Normalize L2 distance
                            distance_scores.append(dist_score)
                            
                        valid_indices.append(idx)
                    except Exception:
                        continue
            
            if not similarities or not distance_scores:
                return 0.3
                
            # Weighted confidence calculation
            avg_similarity = np.mean(similarities)
            avg_distance = np.mean(distance_scores)
            coverage = len(valid_indices) / len(indices[0])
            
            confidence = (0.5 * avg_similarity) + (0.3 * avg_distance) + (0.2 * coverage)
            return min(1.0, max(0.3, confidence * 1.2))

        # --- Robust Answer Generation ---
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def generate_with_retry(model, prompt):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 1000
                    },
                    stream=True
                )
                return response
            except Exception as e:
                st.toast(f"Retrying... ({str(e)})", icon="🔄")
                raise

        def generate_answer_with_confidence(query, chunks, index):
            if not query or not chunks or index is None:
                return "System not ready - please try again"
                
            try:
                model_name = "gemini-1.5-flash" if "flash" in model_choice.lower() else "gemini-1.5-pro"
                model = genai.GenerativeModel(model_name)
                
                # Safe context retrieval
                query_embed = embedder.encode(query).reshape(1, -1)
                k = min(5, len(chunks))
                distances, indices = index.search(query_embed, k)
                
                # Get chunks with bounds checking
                context_chunks = []
                for idx in indices[0]:
                    if 0 <= idx < len(chunks):
                        context_chunks.append(chunks[idx])
                
                context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found"
                
                # Calculate comprehensive confidence score
                confidence = calculate_context_confidence(
                    query_embed, 
                    chunks,  # Pass all chunks for on-demand encoding
                    indices,
                    distances
                )
                confidence_percent = min(99, max(30, int(confidence * 100)))
                
                # Generate answer
                prompt = f"""Analyze this context and answer the question thoroughly:

CONTEXT:
{context}

QUESTION: {query}

Provide a complete answer with reasoning and relevant calculations:"""
                
                response = generate_with_retry(model, prompt)
                full_response = "".join([chunk.text for chunk in response])
                
                # Dynamic confidence display
                if confidence_percent >= 75:
                    conf_color = "#4CAF50"  # Green
                    conf_text = "High confidence"
                elif confidence_percent >= 50:
                    conf_color = "#FFC107"  # Yellow
                    conf_text = "Medium confidence"
                else:
                    conf_color = "#F44336"  # Red
                    conf_text = "Low confidence"
                
                st.markdown(
                    f"""<div style="display: inline-block; padding: 0.35em 0.65em;
                    border-radius: 8px; background-color: {conf_color}; 
                    color: white; font-weight: bold; margin: 12px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <span style="font-size: 0.9em;">{conf_text}:</span> {confidence_percent}%
                    </div>""",
                    unsafe_allow_html=True
                )
                
                return full_response
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                return f"Could not generate answer: {str(e)}"

        # --- Document Processing with Validation ---
        def process_document(uploaded_file):
            if not uploaded_file:
                return None, None
                
            try:
                if uploaded_file.type == "application/pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = " ".join(page.extract_text() or "" for page in pdf.pages)
                else:
                    df = pd.read_csv(uploaded_file)
                    text = df.to_markdown()
                
                if not text.strip():
                    st.error("Document is empty")
                    return None, None
                    
                chunks = text_splitter.split_text(text)
                if not chunks:
                    st.error("Could not extract text chunks")
                    return None, None
                    
                embeddings = np.array(embedder.encode(chunks, batch_size=32)).astype('float32')
                index = faiss.IndexFlatIP(embeddings.shape[1])
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                
                return chunks, index
                
            except Exception as e:
                st.error(f"Document processing failed: {str(e)}")
                return None, None

        # --- UI ---
        st.header("📊 Smart Document Analyst")
        uploaded_file = st.file_uploader(
            "Upload PDF/CSV", 
            type=["pdf", "csv"],
            help="For best results, use documents with clear structure"
        )

        if uploaded_file:
            if "rag_data" not in st.session_state:
                st.session_state.rag_data = process_document(uploaded_file)
            chunks, index = st.session_state.rag_data

            if chunks and index is not None:
                query = st.text_input(
                    "Ask your question:",
                    placeholder="E.g., 'What were the top expenses by percentage?'",
                    key="query_input"
                )
                
                if st.button("Analyze", type="primary"):
                    if not query.strip():
                        st.warning("Please enter a question")
                    else:
                        with st.spinner("Analyzing with confidence scoring..."):
                            answer = generate_answer_with_confidence(query, chunks, index)
                            st.markdown(f"**Answer:**\n\n{answer}")
                            
                            # Show sources with bounds checking
                            with st.expander("View sources used"):
                                try:
                                    query_embed = embedder.encode(query).reshape(1, -1)
                                    _, indices = index.search(query_embed, min(3, len(chunks)))
                                    for i, idx in enumerate(indices[0]):
                                        if 0 <= idx < len(chunks):
                                            st.text_area(f"Source {i+1}", chunks[idx], height=150)
                                except Exception as e:
                                    st.error(f"Could not retrieve sources: {str(e)}")
            else:
                st.error("Document processing failed - please try another file")
    else:
        st.warning("Please enter your Gemini API key")



