import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import LinearRegression, LassoCV, LassoLarsCV, LarsCV, Lasso, OrthogonalMatchingPursuitCV, LassoLars, OrthogonalMatchingPursuit, ElasticNetCV, ElasticNet, TweedieRegressor, HuberRegressor, RANSACRegressor, BayesianRidge, Ridge, LassoLarsIC, Lars, PassiveAggressiveRegressor, SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import expon, reciprocal, uniform, randint
from scipy.stats import loguniform
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import functions  # Custom functions.py file
import pickle
import lazypredict
import LazyRegressor


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

    # Define data as features + target
    data = data[features + [target_column]]

    # Feature Selection/Extraction Section
    st.header("Step 4: Feature Selection/Extraction Options")

    # Checkbox for feature selection/extraction
    feature_reduction = st.checkbox("Reduce the number of features")

    if feature_reduction:
        # Radio button to choose between feature extraction and feature selection
        reduction_method = st.radio("Choose reduction method", ["Feature Extraction", "Feature Selection"])

        if reduction_method == "Feature Extraction":
            # Add code for feature extraction method options (e.g., PCA, t-SNE)
            extraction_method = st.selectbox("Choose extraction method", ["PCA", "Time Series"])

            if extraction_method == "PCA":
                # Input the variance percentage to keep
                variance_percentage = st.slider("Select the variance percentage to keep", 70.0, 100.0, 98.0, step=1.0)

                # Call the PCA function from the functions file
                reduced_data, total_cols_before, total_cols_after, cum_var = functions.perform_pca(data, target_column, categorical_columns, variance_percentage)
               
                # Create a container to display information about PCA
                with st.expander("PCA Results"):
                
                    # Display a preview of the reduced data with clear column headers
                    st.dataframe(reduced_data.head(), width=700, height=200)  # Adjust width and height as needed

                    # Plot the cumulative explained variance ratio
                    functions.plot_cumulative_variance(cum_var,variance_percentage)
                    
                    # Display column count information in a visually distinct way
                    col_count_info = f"""
                    **Number of Features:**
                    - **Before PCA:** {total_cols_before}
                    - **After PCA:** {total_cols_after}
                    """
                    # present the results
                    st.markdown(col_count_info)
                
                # Define data as the reduced number of bands
                data = reduced_data

            elif extraction_method == "Time Series":
                # Call the Time series-based feature extraction function
                reduced_data = functions.time_series_feature_extraction(data, target_column, categorical_columns)
    
                # Display the results or any additional information
                with st.expander("Time Series-Based Feature Extraction Results"):
                    st.dataframe(reduced_data, width=700, height=200)
                
                # Define data as the reduced number of bands
                data = reduced_data

        
        elif reduction_method == "Feature Selection":
            # Add code for feature selection method options (e.g., Recursive Feature Elimination, SelectKBest)
            selection_method = st.selectbox("Choose selection method", ["NDSI", "SelectKBest"])
            if selection_method == "NDSI":
                # Call the NDSI Pearson function
                df_results = functions.NDSI_pearson(data, categorical_columns, target_column)
                st.subheader("NDSI Pearson Results")
                st.dataframe(df_results)
               
                # User input for threshold value
                threshold = st.slider('Threshold', min_value=0.0, max_value=1.0, value=0.4)
                # Set maximum distance for local maxima and minima
                max_distance = st.slider('Max Distance', min_value=1, max_value=50, value=10)
                top_bands_list = functions.display_ndsi_heatmap(df_results,threshold,max_distance)

                # Show the NDSI results
                with st.expander("NDSI Results"):
                    final_ndsi_df = functions.calculate_ndsi(data, top_bands_list)
                    # Show the user the number of columns after NDSI calculation
                    st.info(f"Number of columns after NDSI calculation: {len(final_ndsi_df.columns)}")

                    # Add the categorial columns and target column
                    data = pd.concat([final_ndsi_df, data[categorical_columns], data[target_column]], axis=1)
                    st.dataframe(data)
            

    st.header("Step 5: Data Processing Options")

    # Checkbox for handling missing values
    handle_missing_values = st.checkbox("Handle missing values")
    if handle_missing_values:
        missing_values_option = st.radio("Choose missing values handling method",
                                         ["Replace with average", "Replace with 0", "Delete", "Linear Interpolation"])

        if missing_values_option == "Linear Interpolation":
            # Input a limit for linear interpolation
            interpolation_limit = st.number_input("Enter the limit for linear interpolation", min_value=0, max_value=None,
                                                  value=0, step=1)

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

            # Use bar to 
            with st.spinner('Training ML models'):
                # Apply the function to the data
                models_df = functions.evaluate_regression_models(X_train, X_test, y_train, y_test)

            st.write(models_df)
            
            # Define list of all models
            all_models = models_df.index.tolist()

            # Select the model to use using streamlit selectbox
            model_name = st.selectbox("Select a model", all_models)

            # Create dict for model names and functions
            model_names_dict = {
                "Linear Regression": functions.linear_regression_hyperparam_search,
                "Ridge": functions.ridge_hyperparam_search,
                "Bayesian Ridge": functions.bayesian_ridge_hyperparam_search,
                "Lasso Lars IC": functions.lasso_lars_ic_hyperparam_search,
                "Lars": functions.lars_hyperparam_search,
                "Elastic Net": functions.elastic_net_hyperparam_search,
                "Tweedie Regressor": functions.tweedie_regressor_hyperparam_search,
                "Dummy Regressor": functions.dummy_regressor_hyperparam_search,
                "Huber Regressor": functions.huber_regressor_hyperparam_search,
                "SVR": functions.svr_hyperparam_search,
                "RANSAC Regressor": functions.ransac_regressor_hyperparam_search,
                "Transformed Target Regressor": functions.transformed_target_regressor_hyperparam_search,
                "MLP Regressor": functions.mlp_regressor_hyperparam_search,
                "KNN Regressor": functions.knn_regressor_hyperparam_search,
                "Extra Trees Regressor": functions.extra_trees_regressor_hyperparam_search,
                "Kernel Ridge": functions.kernel_ridge_hyperparam_search,
                "Ada Boost Regressor": functions.ada_boost_regressor_hyperparam_search,
                "Passive Aggressive Regressor": functions.passive_aggressive_regressor_hyperparam_search,
                "Gradient Boosting Regressor": functions.gradient_boosting_regressor_hyperparam_search,
                "SGD Regressor": functions.tune_sgd_regressor,
                "Random Forest Regressor": functions.tune_rf_regressor,
                "Hist Gradient Boosting Regressor": functions.tune_hist_gradient_boosting_regressor,
                "Bagging Regressor": functions.tune_bagging_regressor,
                "LightGBM Regressor": functions.tune_lgbm_regressor,
                "XGBoost Regressor": functions.tune_xgb_regressor}

            # Call the function based on the model name
            model_func = model_names_dict[model_name]

            # Call the function
            best_model, best_params = model_func(X_train, X_test, y_train, y_test)

            # Evaluate the models on the test set using the custom function
            results, model_evaluations = functions.evaluate_model(best_model, X_test, y_test)
            
            # Plot the feature importance
            functions.plot_feature_importance(best_model, X_train, y_train)

            # Plot the partial dependence
            functions.plot_pdp(best_model, X_train, features, target_column)

        except Exception as e:
            st.error(f"Error during model training and evaluation: {str(e)}")
else:
    st.info("Please upload a data file to continue.")
