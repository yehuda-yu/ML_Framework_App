# Import streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import expon, reciprocal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import functions

# Helper function to create a download link for a file
  
# Create a title and a sidebar for the app
st.title("End to end ML Regression Model Builder")
st.header("User Input")

# upload file
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

# Perform EDA on the data after it is uploaded and before the model is executed
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        data = pd.read_excel(uploaded_file)
    with st.expander("EDA Options"):
        # Allow the user to select which EDA actions to apply
        st.subheader("EDA Options")
        handle_missing_values = st.checkbox("Handle missing values by drop Nan", value=False)
        handle_outliers = st.checkbox("Handle outliers by Z-score standardization", value=False)
        normalize_data = st.checkbox("Normalize data", value=False)
        encode_categorical_variables = st.checkbox("Encode categorical variables", value=False)
        #visualize_data = st.checkbox("Visualize data", value=True)
       
        # Perform EDA on the data
        functions.perform_eda(data,handle_missing_values,handle_outliers,normalize_data,encode_categorical_variables)
    with st.expander("Find best regression model"):
        # Allow the user to select a target column and a split percentage from the sidebar
        features = st.multiselect("Select features columns", data.columns.tolist(), default=data.columns.tolist())
        target_column = st.selectbox("Select the target column", data.columns)
        data = data[features + [target_column]]
        split_percentage = st.slider("Select the train-test split percentage", 0.1, 0.9, 0.7)
        
        # Run model queue
        run_model = st.button("Run model")
        
        if run_model:
            # present data
            st.subheader("Data Preview")
            st.write(data.head())
            # Shuffle the data and split it into train and test sets based on the user input
            data = data.sample(frac=1, random_state=42) # shuffle the data
            X = data.drop(target_column, axis=1) # features
            y = data[target_column] # target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percentage, random_state=42) # split the data
        
            # Create regression models
            models = {"Linear Regression": LinearRegression(),
                      "Random Forest": RandomForestRegressor(),
                      "SVM Regression": SVR()}
            
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
            
            # For each model, use a progress bar or another widget to show the hyperparameter search with cross-validation
            st.subheader("Model Training")
            best_models = {}  # store the best models for each type
            best_scores = {}  # store the best scores for each type
            best_params = {}  # store the best parameters for each type
            
            for model_type in models.keys():
                st.write(f"Training {model_type} model...")
                with st.spinner('It can take some time...'):
                    # Perform the randomized search with cross-validation
                    search = RandomizedSearchCV(models[model_type], param_grids[model_type], cv=3, n_iter=10, random_state=42)
                    search.fit(X_train, y_train)
            
                # Print the best parameters and score
                st.write(f"Best parameters for {model_type}: ", search.best_params_)
                st.write(f"Best score for {model_type} on train set: ", search.best_score_)
            
                # Store the best model, score, and parameters
                best_models[model_type] = search.best_estimator_
                best_scores[model_type] = search.best_score_
                best_params[model_type] = search.best_params_
             
            # Evaluate the models on the test set and display the results (MAE, MSE, RMSE, R2, etc.)
            st.subheader("Model Evaluation")
            # Create a DataFrame to store the evaluation results
            results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2", "RPD"])
            
            # Evaluate each model on the test set
            for model_type in best_models.keys():
                st.write(f"Evaluating {model_type} model...")
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

        
            # Display the results as a table
            st.write(results)
        
            # Select the best model based on the lowest RMSE and save it as a pickle file
            best_model_type = results.loc[results["RMSE"].idxmin(), "Model"]  # get the model type with the lowest RMSE
            best_model_type_scalar = best_model_type.iloc[0]  # Convert Series to scalar value (string)
            best_model = best_models[best_model_type_scalar]  # get the best model object

            # Modify the file name to include the model name
            file_name = f"best_model_{best_model_type_scalar}.pkl"
            with open(file_name, "wb") as f:
              pickle.dump(best_model, f)  # save the model as a pickle file with the modified file name

        
            # Allow the user to download the pickle file with a button
            st.subheader("Download Best Model")
            st.markdown(f"Click the button below to download the {best_model_type_scalar} model as a pickle file.")
            if st.button("Download"):
                st.markdown(functions.get_binary_file_downloader_html("best_model_{best_model_type_scalar}.pkl", "Best Model"), unsafe_allow_html=True)
    



