import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import functions  # Custom functions.py file
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Set the font size for regular text
plt.rcParams['font.size'] = 14
# Set the font size for titles
plt.rcParams['axes.titlesize'] = 16
# Set the font size for large titles
plt.rcParams['axes.titlesize'] = 20

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

st.table(data)
"""
# Feature columns selection
st.header("Step 2: Feature Columns Selection")
features = st.multiselect("Select features columns", data.columns.tolist(), default=data.columns.tolist())

# Select target column
st.header("Step 3: Target Column Selection")
target_column = st.selectbox("Select the target column", data.columns)

st.header("Step 4: Data Processing Options")

# Checkbox for handling missing values
handle_missing_values = st.checkbox("Handle missing values")
if handle_missing_values:
    missing_values_option = st.radio("Choose missing values handling method", ["Replace with average", "Replace with 0", "Delete"])

# Checkbox for normalization
normalize_data = st.checkbox("Normalize data")
if normalize_data:
    normalization_method = st.radio("Choose normalization method", ["MinMaxScaler", "StandardScaler"])

# Checkbox for encoding
encode_categorical_variables = st.checkbox("Encode categorical variables")
categorical_columns = []
if encode_categorical_variables:
    categorical_columns = st.multiselect("Select categorical columns for encoding", data.columns)

# Run Model button
run_model = st.button("Run Model")

# Run the model if the button is clicked
if run_model:
    try:
        # Perform data processing
        if handle_missing_values:
            if missing_values_option == "Replace with average":
                data = data.fillna(data.mean())
            elif missing_values_option == "Replace with 0":
                data = data.fillna(0)
            elif missing_values_option == "Delete":
                data = data.dropna()

        if normalize_data:
            if normalization_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
            elif normalization_method == "StandardScaler":
                scaler = StandardScaler()
                data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        if encode_categorical_variables and categorical_columns:
            encoder = OneHotEncoder(drop='first', sparse=False)
            encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
            data = pd.concat([data, encoded_data], axis=1)
            data = data.drop(categorical_columns, axis=1)

        # Display the processed data
        st.subheader("Processed Data:")
        st.write(data)

        # Present data
        st.subheader("Data Preview")
        st.write(data[features + [target_column]].head())

        # Shuffle the data and split it into train and test sets based on the user input
        data = data.sample(frac=1, random_state=42) # shuffle the data
        X = data.drop(target_column, axis=1) # features
        y = data[target_column] # target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percentage, random_state=42) # split the data
    
        # Create two regression models: Random Forest and Linear Regression
        models = {"Random Forest": RandomForestRegressor(),
                  "SVM Regression": LinearRegression()}
    
        # Define the parameter grids for each model
        param_grids = {"Random Forest": {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                      'max_features': ['auto', 'sqrt'],
                                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                                      'min_samples_split': [2, 5, 10],
                                      'min_samples_leaf': [1, 2, 4],
                                      'bootstrap': [True, False]},
                    "SVM Regression": {'kernel': ['rbf','linear'],
                                       'shrinking': [False,True],
                                        'C': reciprocal(10, 200),
                                        'epsilon': reciprocal(0.1, 1.0),
                                        'coef0': expon(scale=1.0),
                                        'gamma': expon(scale=1.0),
                                        'degree': [1,2,3,4,5,6],
                                        'tol': expon(scale=1e-4)}}
    
        # For each model, use a progress bar or another widget to show the hyperparameter search with cross-validation
        st.subheader("Model Training")
        best_models = {} # store the best models for each type
        best_scores = {} # store the best scores for each type
        best_params = {} # store the best parameters for each type
    
        for model_type in models.keys():
            st.write(f"Training {model_type} model...")
            with st.spinner('its can takes some time...'):
                
    
                # Perform the randomized search with cross-validation
                search = RandomizedSearchCV(models[model_type], param_grids[model_type], cv=3,
                                            n_iter=10 if model_type != "Linear Regression" else 1,
                                            random_state=42)
                search.fit(X_train, y_train)
    
    
            # Print the best parameters and score
            st.write(f"Best parameters for {model_type}: ", search.best_params_)
            st.write(f"Best score for {model_type}: ", search.best_score_)
    
            # Store the best model, score, and parameters
            best_models[model_type] = search.best_estimator_
            best_scores[model_type] = search.best_score_
            best_params[model_type] = search.best_params_
    
        # Evaluate the models on the test set and display the results (MAE, MSE, RMSE, R2, etc.)
        st.subheader("Model Evaluation")
        results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2", "RPD"]) # store the results for each model
        for model_type in best_models.keys():
            st.write(f"Evaluating {model_type} model...")
            # Predict the target variable for the test set
            y_test_pred = best_models[model_type].predict(X_test)
    
            # Calculate MAE, MSE, RMSE, R2, etc.
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_test_pred)
            rpd = y_test.std()/rmse
    
            # Append the results to the dataframe
            results = results.append({"Model": model_type,
                                      "MAE": mae,
                                      "MSE": mse,
                                      "RMSE": rmse,
                                      "R2": r2,
                                      "RPD": rpd}, ignore_index=True)
    
        # Display the results as a table
        st.write(results)
    
        # Select the best model based on the lowest RMSE and save it as a pickle file
        best_model_type = results.loc[results["RMSE"].idxmin(), "Model"] # get the model type with the lowest RMSE
        best_model = best_models[best_model_type] # get the best model object
        pickle.dump(best_model, open("best_model.pkl", "wb")) # save the model as a pickle file
    
        # Allow the user to download the pickle file with a button
        st.subheader("Download Best Model")
        # Helper function to create a download link for a file
        def get_binary_file_downloader_html(bin_file, file_label="File"):
            import base64
            bin_str = base64.b64encode(bin_file.encode()).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Download {file_label}</a>'
            return href
        st.markdown("Click the button below to download the best model as a pickle file.")
        if st.button("Download"):
            st.markdown(functions.get_binary_file_downloader_html("best_model.pkl", "Best Model"), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error during model training and evaluation: {str(e)}")

    """
