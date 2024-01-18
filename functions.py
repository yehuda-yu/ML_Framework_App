import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px

def replace_missing_with_average(data):
    """Replace missing values with the average of each column."""
    return data.fillna(data.mean())

def replace_missing_with_zero(data):
    """Replace missing values with zero."""
    return data.fillna(0)

def delete_missing_values(data):
    """Delete rows containing missing values."""
    return data.dropna()

def normalize_data_minmax(data):
    """Normalize data using Min-Max scaling."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns)

def normalize_data_standard(data):
    """Standardize data using Z-score standardization."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return pd.DataFrame(standardized_data, columns=data.columns)

def encode_categorical_onehot(data):
    """One-hot encode categorical variables."""
    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
    data = pd.concat([data, encoded_data], axis=1)
    data = data.drop(categorical_columns, axis=1)
    return data

def encode_categorical_label(data):
    """Label encode categorical variables."""
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column])
    return data

def train_models(models, param_grids, X_train, y_train):
    try:
        best_models = {}  # Store the best models for each type
        best_scores = {}  # Store the best scores for each type
        best_params = {}  # Store the best parameters for each type

        for model_type in models.keys():
            st.write(f"Training {model_type} model...")
            # Perform the randomized search with cross-validation
            search = RandomizedSearchCV(models[model_type], param_grids[model_type], cv=3, n_iter=10, random_state=42)
            search.fit(X_train, y_train)

            # Store the best model, score, and parameters
            best_models[model_type] = search.best_estimator_
            best_scores[model_type] = round(search.best_score_, 2)
            best_params[model_type] = search.best_params_

        return best_models, best_scores, best_params

    except Exception as e:
        st.error(f"An error occurred while training models: {e}")
        return None, None, None

def evaluate_models(best_models, X_test, y_test):
    try:
        # Create a DataFrame to store the evaluation results
        results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2", "RPD"])

        # Evaluate each model on the test set and store the results in a dictionary
        model_evaluations = {}
        for model_type, model in best_models.items():
            # Predict the target variable for the test set
            y_test_pred = model.predict(X_test)

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

        return results, model_evaluations

    except Exception as e:
        st.error(f"An error occurred while evaluating models: {e}")
        return None, None


def plot_scatter_subplots(model_evaluations):
    try:
        fig = sp.make_subplots(rows=1, cols=len(model_evaluations), 
                               subplot_titles=list(model_evaluations.keys()))

        for i, (model_type, evaluation) in enumerate(model_evaluations.items()):
            scatter_trace = go.Scatter(x=evaluation["y_test_pred"], 
                                       y=evaluation["y_test"],
                                       mode='markers',
                                       marker=dict(color='#2a9d8f', line=dict(color='black', width=1)),
                                       name=model_type)

            reference_line = go.Scatter(x=[min(evaluation["y_test"]), max(evaluation["y_test"])],
                                        y=[min(evaluation["y_test"]), max(evaluation["y_test"])],
                                        mode='lines',
                                        line=dict(color='black', dash='dash'),
                                        showlegend=False)

            fig.add_trace(scatter_trace, row=1, col=i+1)
            fig.add_trace(reference_line, row=1, col=i+1)

            fig.update_xaxes(title_text="Predictions (test set)", row=1, col=i+1)
            fig.update_yaxes(title_text="True Values (test set)", row=1, col=i+1)

        fig.update_layout(title_text="Scatter Subplots",
                          margin=dict(l=0, r=0, t=60, b=0))

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting scatter subplots: {e}")
        


def plot_feature_importance(best_models, X_train, y_train, model_type_to_title=None):
    try:
        if model_type_to_title is None:
            model_type_to_title = {
                "Linear Regression": "Linear Regression",
                "Random Forest": "Random Forest",
                "SVM Regression": "SVM Regression"
            }

        for i, (model_type, model) in enumerate(best_models.items()):
            fig = go.Figure()

            if hasattr(model, 'feature_importances_'):  # For Random Forest
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                names = [X_train.columns[i] for i in indices]
                importance_values = [importances[i] for i in indices]
                total_importance = np.sum(importance_values)

                fig.add_trace(go.Pie(labels=names, values=importance_values, 
                                     textinfo='label+percent', hole=0.3,
                                     title=model_type_to_title.get(model_type, model_type)))

            else:  # For SVM Regression and other models
                result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
                importances = result.importances_mean
                indices = np.argsort(importances)[::-1]
                names = [X_train.columns[i] for i in indices]
                importance_values = [importances[i] for i in indices]
                total_importance = np.sum(importance_values)

                fig.add_trace(go.Pie(labels=names, values=importance_values, 
                                     textinfo='label+percent', hole=0.3,
                                     title=model_type_to_title.get(model_type, model_type)))

            fig.update_layout(title_text="Feature Importance",
                              margin=dict(l=0, r=0, t=60, b=0))

            st.write(model_type_to_title.get(model_type, model_type))
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting feature importance: {e}")
