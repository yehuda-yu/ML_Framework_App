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
  
def perform_eda(data, handle_missing_values, handle_outliers, normalize_data, encode_categorical_variables):
    try:
        with st.spinner('Performing EDA...'):
            # Handle missing values
            if handle_missing_values:
                data = data.dropna()  # drop rows with missing values

            # Handle outliers using the Z-score method
            if handle_outliers:
                z_scores = (data - data.mean()) / data.std()
                data = data[(z_scores < 3).all(axis=1)]

            # Normalize the data using min-max scaling
            if normalize_data:
                scaler = MinMaxScaler()
                data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

            # Encode categorical variables using one-hot encoding
            if encode_categorical_variables:
                categorical_columns = data.select_dtypes(include=['object']).columns
                encoder = OneHotEncoder()
                encoded_data = encoder.fit_transform(data[categorical_columns])
                data = pd.concat([data.drop(categorical_columns, axis=1), pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())], axis=1)

        return data

    except Exception as e:
        st.error(f"An error occurred while performing EDA: {e}")
        return None

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
        fig, axes = plt.subplots(1, 3, figsize=(18, 9))

        for i, (model_type, evaluation) in enumerate(model_evaluations.items()):
            ax = axes[i]
            ax.scatter(evaluation["y_test_pred"], evaluation["y_test"], alpha=0.8, color='#2a9d8f', edgecolors='black')
            ax.plot([evaluation["y_test"].min(), evaluation["y_test"].max()],
                    [evaluation["y_test"].min(), evaluation["y_test"].max()],
                    'k--', lw=2)
            ax.set_xlabel('Predictions (y_test_pred)')
            ax.set_ylabel('True Values (y_test)')
            ax.set_title(model_type)

        plt.tight_layout()
        plt.show()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting scatter subplots: {e}")

def plot_scatter_subplots(model_evaluations):
    try:
        fig = sp.make_subplots(rows=1, cols=3, subplot_titles=list(model_evaluations.keys()), shared_yaxes=True,
                               horizontal_spacing=0.1, vertical_spacing=0.2)

        for i, (model_type, evaluation) in enumerate(model_evaluations.items()):
            fig.add_trace(
                go.Scatter(
                    x=evaluation["y_test_pred"],
                    y=evaluation["y_test"],
                    mode='markers',
                    marker=dict(color='#2a9d8f', opacity=0.8, line=dict(color='black', width=1)),
                    showlegend=False
                ),
                row=1,
                col=i + 1
            )
            fig.add_trace(
                go.Scatter(
                    x=[evaluation["y_test"].min(), evaluation["y_test"].max()],
                    y=[evaluation["y_test"].min(), evaluation["y_test"].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=2),
                    showlegend=False
                ),
                row=1,
                col=i + 1
            )
            fig.update_xaxes(title_text="Predictions (y_test_pred)", row=1, col=i + 1)
            fig.update_yaxes(title_text="True Values (y_test)", row=1, col=i + 1)

        fig.update_layout(title_text="Scatter Subplots", title_x=0.5, width=1000, height=500)
        fig.show()
        # Use st.write() instead of st.pyplot() since Plotly figures are not directly supported by st.pyplot().
        # You can also use st.plotly_chart() to show the plotly figure if you are using Streamlit >= 1.0.0.
        # st.plotly_chart(fig)

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

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, (model_type, model) in enumerate(best_models.items()):
            if hasattr(model, 'feature_importances_'):  # For Random Forest
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                names = [X_train.columns[i] for i in indices]
                importance_values = [importances[i] for i in indices]
                total_importance = np.sum(importance_values)

                ax = axes[i]
                ax.pie(importance_values, labels=names, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', startangle=90)
                ax.set_title(model_type_to_title.get(model_type, model_type))
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                # Add a center circle to make it look like a donut chart
                center_circle = plt.Circle((0, 0), 0.70, fc='white')
                ax.add_artist(center_circle)

            else:  # For SVM Regression and other models
                result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
                importances = result.importances_mean
                indices = np.argsort(importances)[::-1]
                names = [X_train.columns[i] for i in indices]
                importance_values = [importances[i] for i in indices]
                total_importance = np.sum(importance_values)

                ax = axes[i]
                ax.pie(importance_values, labels=names, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '', startangle=90)
                ax.set_title(model_type_to_title.get(model_type, model_type))
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                # Add a center circle to make it look like a donut chart
                center_circle = plt.Circle((0, 0), 0.70, fc='white')
                ax.add_artist(center_circle)

        plt.tight_layout()
        plt.show()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting feature importance: {e}")
