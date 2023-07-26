# Import streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Set the font size for regular text
plt.rcParams['font.size'] = 14
# Set the font size for titles
plt.rcParams['axes.titlesize'] = 16
# Set the font size for large titles
plt.rcParams['axes.titlesize'] = 20
import functions  # Custom functions.py file

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import expon, reciprocal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Create a title for the app
st.title("End to End ML Regression Model Builder")

# Step 1: Upload Data
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

# Perform EDA on the data after it is uploaded and before the model is executed
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        data = pd.read_excel(uploaded_file)

    # Inform the user about successful data upload
    st.success("Data uploaded successfully!")

    # Step 2: Exploratory Data Analysis (EDA) Options
    st.header("Step 2: Exploratory Data Analysis (EDA) Options")
    handle_missing_values = st.checkbox("Handle missing values by drop Nan", value=False)
    handle_outliers = st.checkbox("Handle outliers by Z-score standardization", value=False)
    normalize_data = st.checkbox("Normalize data", value=False)
    encode_categorical_variables = st.checkbox("Encode categorical variables", value=False)

    functions.perform_eda(data, handle_missing_values, handle_outliers, normalize_data, encode_categorical_variables)

    # Step 3: Find best regression model
    st.header("Step 3: Find best regression model")
    features = st.multiselect("Select features columns", data.columns.tolist(), default=data.columns.tolist())
    target_column = st.selectbox("Select the target column", data.columns)
    data = data[features + [target_column]]
    split_percentage = st.slider("Select the train-test split percentage", 0.1, 0.9, 0.7)
    run_model = st.button("Run Model")

    if run_model:
        # Present data
        st.subheader("Data Preview")
        st.write(data.head())

        # Shuffle the data and split it into train and test sets based on the user input
        data = data.sample(frac=1, random_state=42)  # Shuffle the data
        X = data.drop(target_column, axis=1)  # Features
        y = data[target_column]  # Target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_percentage, random_state=42)

        # Create regression models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "SVM Regression": SVR()
        }

        # Hyperparameter grids for RandomizedSearchCV
        param_grids = {
            "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
            "Random Forest": {
                'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            "SVM Regression": {'kernel': ['rbf', 'linear'],
                    'shrinking': [False, True],
                    'C': reciprocal(10, 200),
                    'epsilon': reciprocal(0.1, 1.0),
                    'coef0': expon(scale=1.0),
                    'gamma': expon(scale=1.0),
                    'degree': [1, 2, 3, 4, 5, 6],
                    'tol': expon(scale=1e-4)},
        }
        st.header("Step 4: Model Training")
        best_models, best_scores, best_params = functions.train_models(models, param_grids, X_train, y_train)

        # Step 5: Model Evaluation
        st.header("Step 5: Model Evaluation")
        results, model_evaluations = functions.evaluate_models(best_models, X_test, y_test)
        
        # Step 6: Scatter Plots and table for Model Evaluation
        functions.plot_scatter_subplots(model_evaluations)
        st.table(results.set_index("Model"))

        # Step 7: Feature Importance
        st.header("Step 7: Feature Importance")
        model_type_to_title = {"Linear Regression": "Linear Model",
            "Random Forest": "Random Forest",
            "SVM Regression": "SVM Model"
        }
        functions.plot_feature_importance(best_models, X_train, y_train, model_type_to_title)

        # Download Best Model
        st.header("Step 8: Download Best Model")
        selected_model = st.selectbox("Select the model to download", results["Model"])
        best_model = best_models[selected_model]
        file_name = f"best_model_{selected_model}.pkl"
        
        # Save the selected model as a pickle file
        with open(file_name, "wb") as f:
            pickle.dump(best_model, f)
        
        # Allow the user to download the selected pickle file with a button
        download_button = st.button("Download Best Model")
        if download_button:
            st.download_button(label="Click here to download the best model",
                               data=open(file_name, "rb").read(),
                               file_name=file_name,
                               mime="application/octet-stream")
