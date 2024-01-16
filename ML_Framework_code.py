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

# Data processing options
data = st.session_state.data if 'data' in st.session_state else None  # Initialize data variable using session_state

# Perform EDA on the data after it is uploaded and before the model is executed
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        data = pd.read_excel(uploaded_file)
    st.session_state.data = data  # Save data in session_state

st.header("Step 2: Data Processing Options")

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

# Feature columns selection
st.header("Step 3: Feature Columns Selection")
features = st.multiselect("Select features columns", data.columns.tolist(), default=data.columns.tolist())

# Select target column
st.header("Step 4: Target Column Selection")
target_column = st.selectbox("Select the target column", data.columns)

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
        data = data.sample(frac=1, random_state=42)  # Shuffle the data
        X = data[features]  # Features
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
        st.header("Step 4: Model Training")
        best_models, best_scores, best_params = functions.train_models(models, param_grids, X_train, y_train)

        # Step 5: Model Evaluation
        st.header("Step 5: Model Evaluation")
        results, model_evaluations = functions.evaluate_models(best_models, X_test, y_test)
        
        # Scatter Plots and table for Model Evaluation
        functions.plot_scatter_subplots(model_evaluations)
        st.dataframe(results.set_index("Model"))

        # Step 6: Feature Importance
        st.header("Step 6: Feature Importance")
        model_type_to_title = {"Linear Regression": "Linear Model",
            "Random Forest": "Random Forest",
            "SVM Regression": "SVM Model"
        }

        # Plot feature importance
        functions.plot_feature_importance(best_models, X_train, y_train)
        
        # Get the index of the model with the highest R2 score
        best_r2_index = results["R2"].idxmax()
        best_model_name = results.iloc[best_r2_index]["Model"]
        file_name = f"best_model_{best_model_name}.pkl"
        
        # Save the best model as a pickle file
        best_model = best_models[best_model_name]
        with open(file_name, "wb") as f:
            pickle.dump(best_model, f)
        
        # Provide a download link for the best model
        st.download_button(label="Download the best model with highest R2",
                           data=open(file_name, "rb").read(),
                           file_name=file_name,
                           mime="application/octet-stream")
