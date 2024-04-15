
# Comprehensive Guide to the Kobe Shot Prediction Model

This guide consolidates various Python scripts and modules used throughout the Kobe Shot Prediction project. The project utilizes a range of data processing, model training, evaluation techniques, and deployment strategies to predict basketball shot outcomes.

## Prerequisites
Make sure to install all required libraries:
```bash
pip install pandas numpy scikit-learn mlflow pycaret seaborn matplotlib statsmodels streamlit
```

## Scripts and Functionality

### Data Processing and Visualization
- **Data Loading and Preparation**: Uses Pandas for data handling, splitting datasets into training and test sets, and preprocessing data with scikit-learn's `StandardScaler`.
- **Visualization**: Utilizes Matplotlib and Seaborn for generating various plots such as histograms, correlation matrices, and ROC curves.

### Model Training and Evaluation
- **Model Training**: Employs PyCaret to compare models, perform hyperparameter tuning, and select the best model based on accuracy.
- **Model Validation**: Implements cross-validation techniques to assess model reliability and uses validation curves to visualize performance.

### Model Deployment and Monitoring
- **MLflow Tracking**: Sets up MLflow for experiment tracking, parameter logging, and model versioning.
- **Streamlit Dashboard**: Deploys a Streamlit application to interactively display model predictions, performance metrics, and allows downloading model artifacts directly from MLflow.

### Alert System
- **Performance Monitoring**: Includes functionality to monitor model performance and trigger alerts if the performance metrics fall below a predefined threshold.

## Code Examples

### Model Monitoring and Alerting
```python
def alarm(data_monitoring, min_eff_alarm, metric_t=0.35):
    # Calculates and logs model performance metrics
    # Determines if the model needs retraining based on the alarm threshold
```

### Streamlit Dashboard for Model Insights
```python
import streamlit as st

# Set up and configuration
st.set_page_config(page_title="Kobe Shot Prediction Dashboard")

# Load model and data, display metrics and predictions
st.write("Welcome to the Kobe Shot Prediction Dashboard.")
```

### MLflow for Model Tracking
```python
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI and experiment details
mlflow.set_tracking_uri("sqlite:///mlruns.db")
experiment_name = "Monitoramento"
```

## Running the Project
To run the full pipeline, execute the main script:
```bash
python main.py
```

To view the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

Ensure to follow the instructions in `README.md` for detailed setup and execution guidelines.

## Conclusion
This comprehensive guide and accompanying scripts provide a robust framework for predicting basketball shot outcomes, tracking model performance, and offering interactive insights through a web-based dashboard.

## Additional MLflow Commands

To interact with the MLflow tracking server and serve models, use the following commands:

1. **Launch MLflow UI**:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlruns.db
   ```

2. **Set MLflow Tracking URI** (Environment Variable):
   ```bash
   set MLFLOW_TRACKING_URI=sqlite:///mlruns.db
   echo MLFLOW_TRACKING_URI
   ```

3. **Serve the Model**:
   ```bash
   mlflow models serve -m "models:/modelo_kobe@staging" --no-conda -p 5000
   ```

These commands facilitate viewing the MLflow dashboard, setting up the tracking URI, and serving the model for predictions.
