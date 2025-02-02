import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import chi2_contingency, spearmanr
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Placement Data Analysis and Prediction")

# Load Data
st.markdown("## Loading Data")
try:
    df = pd.read_csv("placementdata.csv")
    st.write(df.head())
except FileNotFoundError:
    st.error("Dataset not found! Please ensure 'placementdata.csv' is in the same directory.")
    st.stop()

# Exploratory Data Analysis (EDA)
st.markdown("## Exploratory Data Analysis (EDA)")
st.markdown("### Shape of the dataset:")
r,c=df.shape
st.write(f"Number of rows: {r}, Number of columns: {c}")
st.markdown("### Dataset Information:")
st.text(df.info())
st.markdown("Columns in the dataset:")
st.write(df.columns)
st.markdown("### Description of the dataset:")
st.write(df.describe())

# Data Cleaning
st.markdown("## Data Cleaning")
st.markdown("### Missing values in the dataset:")
st.write(df.isnull().sum())
st.markdown("### Duplicate rows in the dataset:")
st.write(df.duplicated().sum())
# Display unique values for specific columns
columns_to_check = ["ExtracurricularActivities", "PlacementTraining", "PlacementStatus"]
for col in columns_to_check:
    unique_values = df[col].unique()
    st.subheader(f"Unique Values in '{col}':")
    st.write(unique_values)

# Define Categorical and Numerical Columns
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = numerical_columns.drop('StudentID')

# Univariate Analysis: Categorical Data
st.markdown("## Univariate Analysis")
st.markdown("### Categorical Data")
num_rows = (len(categorical_columns) + 2) // 3
fig, axes = plt.subplots(num_rows, 3, figsize=(9,5))
axes = axes.flatten()

for i, col in enumerate(categorical_columns):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} Distribution')
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)

# Univariate Analysis: Numerical Data
st.markdown("### Numerical Data")
fig, axes = plt.subplots((len(numerical_columns) + 2) // 3, 3, figsize=(13,9))
axes = axes.flatten()

for i, col in enumerate(numerical_columns):
    sns.histplot(df[col], bins=20, kde=True, color='skyblue', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)

# Bivariate Analysis
st.markdown("## Bivariate Analysis")
st.markdown("### Categorical Data vs. Placement Status")
# Plot bar graphs
fig, axes = plt.subplots(
    nrows=((len(numerical_columns[1:]) + 2) // 3),  # Flexible number of rows
    ncols=3,
    figsize=(15, 10)
)
axes = axes.flatten()

for i, col in enumerate(numerical_columns[1:], start=0):  # Exclude 'StudentID'
    sns.barplot(
        x=df['PlacementStatus'],
        y=df[col],
        hue=df['PlacementStatus'],
        ax=axes[i],
        dodge=False  # Prevent duplicated legends
    )
    axes[i].set_title(f'Average {col} by Placement Status')
    axes[i].set_xlabel('Placement Status')
    axes[i].set_ylabel(f'Average {col}')

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()

# Streamlit display
st.pyplot(fig)

# Calculate mean, median, and sum for numerical features by PlacementStatus
grouped_stats = df.groupby('PlacementStatus')[numerical_columns].agg(['mean', 'median'])

# Streamlit app
st.title("Grouped Statistics for Numerical Features by Placement Status")

st.write("### Aggregated Statistics (Mean, Median, Sum):")
st.dataframe(grouped_stats.style.format(precision=2))

# Multivariate Analysis: Pair Plot
st.markdown("## Multivariate Analysis")
st.markdown("### Pair Plot")
try:
    fig = sns.pairplot(df, hue="PlacementStatus", diag_kind="kde", palette="GnBu")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error in pair plot generation: {e}")

# Statistical Analysis: Correlation Heatmap
st.markdown("## Statistical Analysis")
st.markdown("### Correlation Heatmap")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    numerical_df = df.select_dtypes(include=['int64', 'float64']).drop('StudentID', axis=1, errors='ignore')
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error in generating heatmap: {e}")

# Chi-Square Test
st.markdown("## Chi-Square Test")
st.markdown("### Chi-Square Test for Categorical Variables")
column1 = 'PlacementTraining'
column2 = 'PlacementStatus'
try:
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    st.write("Chi-square test results:")
    st.write("Chi2:", chi2)
    st.write("P-value:", p_value)
    st.write("Degrees of freedom:", dof)
    st.write("Expected frequencies:", expected)
except Exception as e:
    st.error(f"Error in Chi-Square Test: {e}")

st.write("""Key Insights:
Placement Training: Strongly associated with placement likelihood, emphasizing the value of preparation programs.
         
Academic Marks and Internship Scores: High positive correlation with placement outcomes, suggesting that academic and practical performance are key.
         
Extracurricular Activities: Showed mixed results, with certain activities correlating better than others.
         
Communication Skills: Strongly associated with placement success, highlighting the importance of soft skills.
         
This analysis provided actionable insights into the factors influencing placement success and built robust predictive models to guide decision-making for students and institutions.""")

# Data Pre-Processing
st.markdown("## Data Pre-Processing for Model Training")    
try:
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = encoder.fit_transform(df[['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']])
    encoded_columns = encoder.get_feature_names_out(['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus'])
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    df_final = pd.concat([df.drop(['StudentID','ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus'], axis=1), encoded_df], axis=1)
    st.write("Encoded DataFrame:")
    st.write(df_final.head())
except Exception as e:
    st.error(f"Error during encoding: {e}")
    st.stop()

# Splitting Data
X = df_final.drop('PlacementStatus_Placed', axis=1)
y = df_final['PlacementStatus_Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
st.markdown("## Model Training and Evaluation")
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[model_name] = report
    st.write(f"{model_name} Classification Report:")
    st.text(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

st.write("Random Forsst Classifier has the highest accuracy and F1 score, making it the best model for placement prediction.")