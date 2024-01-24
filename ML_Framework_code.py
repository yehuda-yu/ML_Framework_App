import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from scipy.stats import expon, reciprocal
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import functions  # Custom functions.py file
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from sklearn.inspection import PartialDependenceDisplay

# Set the font size for regular text
plt.rcParams['font.size'] = 14
# Set the font size for titles
plt.rcParams['axes.titlesize'] = 16
# Set the font size for large titles
plt.rcParams['axes.titlesize'] = 20
# set the palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#2a9d8f', '#e76f51', '#f4a261', '#738bd7', '#d35400', '#a6c7d8'])

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

   # Feature columns selection
    st.header("Step 2: Feature Columns Selection")
    features = st.multiselect("Select features columns", data.columns.tolist(), default=data.columns.tolist())
    
    # Check if there are categorical columns
    has_categorical_columns = st.checkbox("Are there categorical columns?")
    categorical_columns = []
    
    if has_categorical_columns:
        categorical_columns = st.multiselect("Select categorical columns", features)
    
    # Select target column
    st.header("Step 3: Target Column Selection")
    target_column = st.selectbox("Select the target column", data.columns)

    # Feature Selection/Extraction Section
    st.header("Step 4: Feature Selection/Extraction Options")
    
    # Checkbox for feature selection/extraction
    feature_reduction = st.checkbox("Reduce the number of features")
    
    if feature_reduction:
        # Radio button to choose between feature extraction and feature selection
        reduction_method = st.radio("Choose reduction method", ["Feature Extraction", "Feature Selection"])
    
        if reduction_method == "Feature Extraction":
            # Add code for feature extraction method options (e.g., PCA, t-SNE)
            extraction_method = st.selectbox("Choose extraction method", ["PCA", "t-SNE"])
            if extraction_method == "PCA":
                # Input the variance percentage to keep
                variance_percentage = st.slider("Select the variance percentage to keep", 70.0, 100.0, 95.0, step=1.0)
                
                # Call the PCA function from the functions file
                reduced_data, total_cols_before, total_cols_after = functions.perform_pca(data, target_column, categorical_columns, variance_percentage)
                
                # Display the processed data
                st.subheader("Processed Data after PCA:")
                st.write(reduced_data)
    
                # Display the total number of columns before and after PCA
                st.subheader("Total Number of Columns:")
                st.write(f"Before PCA: {total_cols_before}, After PCA: {total_cols_after}")

        elif reduction_method == "Feature Selection":
            # Add code for feature selection method options (e.g., Recursive Feature Elimination, SelectKBest)
            selection_method = st.selectbox("Choose selection method", ["Recursive Feature Elimination", "SelectKBest"])

    st.header("Step 5: Data Processing Options")
    
    # Checkbox for handling missing values
    handle_missing_values = st.checkbox("Handle missing values")
    if handle_missing_values:
        missing_values_option = st.radio("Choose missing values handling method", ["Replace with average", "Replace with 0", "Delete", "Linear Interpolation"])
    
        if missing_values_option == "Linear Interpolation":
            # Input a limit for linear interpolation
            interpolation_limit = st.number_input("Enter the limit for linear interpolation", min_value=0, max_value=None, value=0, step=1)

    
    # Checkbox for normalization
    normalize_data = st.checkbox("Normalize data")

    if normalize_data:
        normalization_method = st.radio("Choose normalization method", ["MinMaxScaler", "StandardScaler"])

    # Checkbox for encoding
    encode_categorical_variables = st.checkbox("Encode categorical variables")
    categorical_encoding_method = None
    
    if encode_categorical_variables and has_categorical_columns:
        categorical_encoding_method = st.radio("Choose categorical encoding method", ["OneHotEncoder", "LabelEncoder"])
    
    
    # Allow the user to select a split percentage from slider
    split_percentage = st.slider("Select the train-test split percentage", 0.1, 0.9, 0.7)
    
    # Run Model button
    run_model = st.button("Run Model")
    
    # Run the model if the button is clicked
    if run_model:
        try:
            # define data as features + target
            data = data[features + [target_column]]
            
            # Perform data processing
            if handle_missing_values:
                if missing_values_option == "Replace with average":
                    data = data.fillna(data.mean())
                elif missing_values_option == "Replace with 0":
                    data = data.fillna(0)
                elif missing_values_option == "Delete":
                    data = data.dropna()
                elif missing_values_option == "Linear Interpolation":
                    # Perform linear interpolation with the specified limit
                    data = data.interpolate(limit=interpolation_limit)
    
            # Perform encoding
            if has_categorical_columns:
                # Define numerical_columns based on user selection and numerical dtypes
                numerical_columns = list(set(data.columns) - set(categorical_columns))  # Exclude categorical columns
                numerical_columns = list(set(numerical_columns).intersection(data.select_dtypes(include=['number']).columns))  # Ensure they have numerical dtypes
                
                if encode_categorical_variables and categorical_encoding_method and categorical_columns:
                    if categorical_encoding_method == "OneHotEncoder":
                        # Apply one-hot encoding using pandas get_dummies
                        encoded_data = pd.get_dummies(data[categorical_columns], columns=categorical_columns, dtype=float)
                        data = pd.concat([data.drop(categorical_columns, axis=1), encoded_data], axis=1)
                    elif categorical_encoding_method == "LabelEncoder":
                        label_encoder = LabelEncoder()
                        for col in categorical_columns:
                            data[col] = label_encoder.fit_transform(data[col])
            
            # Perform normalization only on numerical columns
            if normalize_data:
                if normalization_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
                elif normalization_method == "StandardScaler":
                    scaler = StandardScaler()
                    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        
            # Display the processed data
            st.subheader("Processed Data:")
            st.write(data)
    
    
            # Shuffle the data and split it into train and test sets based on the user input
            data = data.sample(frac=1, random_state=42) # shuffle the data
            X = data.drop(target_column, axis=1) # features
            y = data[target_column] # target
            # Convert column names to strings
            X.columns = X.columns.astype(str)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percentage, random_state=42) # split the data
    
            # Create two regression models: Random Forest and Linear Regression
            models = {"Random Forest": RandomForestRegressor(),
                      "SVM Regression": SVR(),
                      "Linear Regression": LinearRegression()}
    
            # Define the parameter grids for each model
            param_grids = {
                "Random Forest": {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                                  'max_features': ['auto', 'sqrt'],
                                  'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                                  'min_samples_split': [2, 5, 10],
                                  'min_samples_leaf': [1, 2, 4],
                                  'bootstrap': [True, False]},
                
                "SVM Regression": {'kernel': ['rbf', 'linear'],
                                   'shrinking': [False, True],
                                   'C': reciprocal(10, 200),
                                   'epsilon': reciprocal(0.1, 1.0),
                                   'coef0': expon(scale=1.0),
                                   'gamma': expon(scale=1.0),
                                   'degree': [1, 2, 3, 4, 5, 6],
                                   'tol': expon(scale=1e-4)},
            
                "Linear Regression": {'fit_intercept': [True, False],
                                      'copy_X': [True, False],
                                      'n_jobs': [None, 1, 2, 4],
                                      'positive': [True, False],}  
            }
    
            # Training the models using the custom function
            best_models, best_scores, best_params = functions.train_models(models, param_grids, X_train, y_train)
    
            
            # Evaluate the models on the test set using the custom function
            results, model_evaluations = functions.evaluate_models(best_models, X_test, y_test)
        
            # Plot scatter subplots using the custom function
            functions.plot_scatter_subplots(model_evaluations)
    
            # present in table
            st.dataframe(results.set_index('Model'))
        
            # Plot feature importance using the custom function
            functions.plot_feature_importance(best_models, X_train, y_train)
    
            # Display PDP graphs for selected feature
            st.header("Partial Dependence Plots (PDP)")
    
            selected_feature = st.selectbox("Select feature to visualize", X_train.columns)
    
            if selected_feature:
                # Call the function to plot PDP with specified colors
                functions.plot_pdp(best_models, X_train, [selected_feature], target_column)

            # Save session state
            st.session_state.selected_feature = selected_feature
    
        except Exception as e:
            st.error(f"Error during model training and evaluation: {str(e)}")
else:
    st.info("Please upload a data file to continue.")

        # Allow the user to download the pickle file with a button
        #st.subheader("Download Best Model")
        # Helper function to create a download link for a file
        #def get_binary_file_downloader_html(bin_file, file_label="File"):
         #   import base64
          #  bin_str = base64.b64encode(bin_file.encode()).decode()
           # href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Download {file_label}</a>'
            #return href
        #st.markdown("Click the button below to download the best model as a pickle file.")
        #if st.button("Download"):
         #   st.markdown(functions.get_binary_file_downloader_html("best_model.pkl", "Best Model"), unsafe_allow_html=True)


