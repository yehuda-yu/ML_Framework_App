# Import streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Helper function to create a download link for a file
# (Put the implementation of the helper function here)

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
            "SVM Regression": {},
        }

        # Model Training
        st.header("Step 4: Model Training")
        best_models = {}  # Store the best models for each type
        best_scores = {}  # Store the best scores for each type
        best_params = {}  # Store the best parameters for each type

        for model_type in models.keys():
            st.write(f"Training {model_type} model...")
            with st.spinner('Training in progress...'):
                # Perform the randomized search with cross-validation
                search = RandomizedSearchCV(models[model_type], param_grids[model_type], cv=3, n_iter=10, random_state=42)
                search.fit(X_train, y_train)

            # Print the best parameters and score
            st.write(f"Best score for {model_type} on the train set: ", round(search.best_score_, 2))

            # Store the best model, score, and parameters
            best_models[model_type] = search.best_estimator_
            best_scores[model_type] = search.best_score_
            best_params[model_type] = search.best_params_

        # Model Evaluation
        st.header("Step 5: Model Evaluation")
        # Create a DataFrame to store the evaluation results
        results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2", "RPD"])

        # Evaluate each model on the test set and store the results in a dictionary
        model_evaluations = {}
        for model_type in best_models.keys():
            # Predict the target variable for the test set
            y_test_pred = best_models[model_type].predict(X_test)
        
            # Calculate MAE, MSE, RMSE, R2, etc.
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_test_pred)
            rpd = y_test.std() / rmse
        
            # Append the results to the dataframe
            results = pd.concat([results, pd.DataFrame({"Model": [model_type],
                                                        "MAE": [mae],
                                                        "MSE": [mse],
                                                        "RMSE": [rmse],
                                                        "R2": [r2],
                                                        "RPD": [rpd]})])
        
            # Store the model evaluation results in the dictionary
            model_evaluations[model_type] = {
                "y_test": y_test,
                "y_test_pred": y_test_pred
            }
        
        # Plot scatter subplots for all models after the loop
        st.header("Step 7: Scatter Plots for Model Evaluation")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (model_type, evaluation) in enumerate(model_evaluations.items()):
            ax = axes[i]
            ax.scatter(evaluation["y_test"], evaluation["y_test_pred"], alpha=0.7, edgecolors='w')
            ax.plot([evaluation["y_test"].min(), evaluation["y_test"].max()],
                    [evaluation["y_test"].min(), evaluation["y_test"].max()],
                    'k--', lw=2)
            ax.set_xlabel('True Values (y_test)')
            ax.set_ylabel('Predictions (y_test_pred)')
            ax.set_title(model_type)
        
        plt.tight_layout()
        st.pyplot(fig)

        # Display the results as a table
        st.write(results)   

        # Download Best Model
        st.header("Step 6: Download Best Model")
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
