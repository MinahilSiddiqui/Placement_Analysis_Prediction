Placement Data Analysis and Prediction
This project implements an end-to-end solution for analyzing placement data and predicting outcomes using machine learning models. It is built using Python and integrates with Streamlit for an interactive user interface.

Features
Interactive Data Analysis:

Load and explore placement data through visualizations and statistical summaries.

Perform univariate, bivariate, and multivariate analyses.

Statistical Insights:

Generate correlation heatmaps.

Perform chi-square tests to understand categorical relationships.

Machine Learning Pipeline:

Preprocess data with encoding, scaling, and train-test splitting.

Train and evaluate models (Logistic Regression, Decision Tree, Random Forest).

Visualize performance metrics with classification reports and confusion matrices.

Tech Stack
Frontend: Streamlit for the web interface.

Backend: Python for data preprocessing and model training.

Visualization: Matplotlib, Seaborn for interactive plots.

Machine Learning: Scikit-learn for modeling and evaluation.

Setup and Installation
Prerequisites
Python 3.8 or later.

Required Python packages (see requirements.txt).

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/placement-prediction.git
cd placement-prediction
Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Ensure the dataset (placementdata.csv) is in the project directory.

Usage
Run the Streamlit application:

bash
'''
streamlit run app.py
'''
Interact with the application:

Load the dataset for exploration.

View EDA insights like distributions and relationships.

Train and evaluate machine learning models.

Analyze the results:

Inspect classification reports and confusion matrices.

View recommendations for the best-performing model.

Project Workflow
Data Loading:

Load the dataset and inspect its structure.

EDA:

Analyze categorical and numerical features using visualizations.

Identify key patterns and trends in placement outcomes.

Statistical Analysis:

Compute correlations and test relationships using chi-square.

Data Preprocessing:

Handle categorical encoding and scaling for numerical features.

Model Training:

Train Logistic Regression, Decision Tree, and Random Forest models.

Evaluate models using classification reports and confusion matrices.

Model Insights:

Random Forest achieves the best accuracy and F1 score for placement prediction.

Key Insights
Placement training significantly increases the likelihood of placement.

Academic performance and internships correlate positively with placement outcomes.

Communication skills are critical for placement success.

Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

For queries, contact Manahil Siddiqui.
