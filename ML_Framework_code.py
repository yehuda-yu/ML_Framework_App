# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:58:32 2023

@author: User
"""

# Import streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Create a title and a sidebar for the app
st.title("Regression Model Builder")
st.sidebar.header("User Input")

# Allow the user to upload a CSV or EXCEL file and display it as a table
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        data = pd.read_excel(uploaded_file)
    st.subheader("Data Preview")
    st.write(data.head())

    # Allow the user to select a target column and a split percentage from the sidebar
    target_column = st.sidebar.selectbox("Select the target column", data.columns)
    split_percentage = st.sidebar.slider("Select the train-test split percentage", 0.1, 0.9, 0.7)

    # Shuffle the data and split it into train and test sets based on the user input
    data = data.sample(frac=1, random_state=42) # shuffle the data
    X = data.drop(target_column, axis=1) # features
    y = data[target_column] # target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percentage, random_state=42) # split the data

    # Create two regression models: Random Forest and Linear Regression
    models = {"Random Forest": RandomForestRegressor(),
              "Linear Regression": LinearRegression()}

    # Define the parameter grids for each model
    param_grids = {"Random Forest": {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                  'max_features': ['auto', 'sqrt'],
                                  'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                                  'min_samples_split': [2, 5, 10],
                                  'min_samples_leaf': [1, 2, 4],
                                  'bootstrap': [True, False]},
                "Linear Regression": {}}

    # For each model, use a progress bar or another widget to show the hyperparameter search with cross-validation
    st.subheader("Model Training")
    best_models = {} # store the best models for each type
    best_scores = {} # store the best scores for each type
    best_params = {} # store the best parameters for each type

    for model_type in models.keys():
        st.write(f"Training {model_type} model...")
        progress_bar = st.progress(0) # create a progress bar

        # Perform the randomized search with cross-validation
        search = RandomizedSearchCV(models[model_type], param_grids[model_type], cv=3,
                                    n_iter=10 if model_type != "Linear Regression" else 1,
                                    random_state=42)
        search.fit(X_train, y_train)

        # Update the progress bar
        progress_bar.progress(100)

        # Print the best parameters and score
        st.write(f"Best parameters for {model_type}: ", search.best_params_)
        st.write(f"Best score for {model_type}: ", search.best_score_)

        # Store the best model, score, and parameters
        best_models[model_type] = search.best_estimator_
        best_scores[model_type] = search.best_score_
        best_params[model_type] = search.best_params_

    # Evaluate the models on the test set and display the results (MAE, MSE, RMSE, R2, etc.)
    st.subheader("Model Evaluation")
    results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2", "Error (%)", "Range Error (%)", "RPD"]) # store the results for each model
    for model_type in best_models.keys():
        st.write(f"Evaluating {model_type} model...")
        # Predict the target variable for the test set
        y_test_pred = best_models[model_type].predict(X_test)

        # Calculate MAE, MSE, RMSE, R2, etc.
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)
        error = 100*rmse/y_test.mean()
        range_error = 100*rmse/(y_test.max()-y_test.min())
        rpd = y_test.std()/rmse

        # Append the results to the dataframe
        results = results.append({"Model": model_type,
                                  "MAE": mae,
                                  "MSE": mse,
                                  "RMSE": rmse,
                                  "R2": r2,
                                  "Error (%)": error,
                                  "Range Error (%)": range_error,
                                  "RPD": rpd}, ignore_index=True)

    # Display the results as a table
    st.write(results)

    # Select the best model based on the lowest RMSE and save it as a pickle file
    best_model_type = results.loc[results["RMSE"].idxmin(), "Model"] # get the model type with the lowest RMSE
    best_model = best_models[best_model_type] # get the best model object
    pickle.dump(best_model, open("best_model.pkl", "wb")) # save the model as a pickle file

    # Allow the user to download the pickle file with a button
    st.subheader("Download Best Model")
    st.markdown("Click the button below to download the best model as a pickle file.")
    if st.button("Download"):
        st.markdown(get_binary_file_downloader_html("best_model.pkl", "Best Model"), unsafe_allow_html=True)

# Helper function to create a download link for a file
def get_binary_file_downloader_html(bin_file, file_label="File"):
    import base64
    bin_str = base64.b64encode(bin_file.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Download {file_label}</a>'
    return href
