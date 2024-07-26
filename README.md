# ShoeAskew_EIS_MC_Predictor_Model
Sets target variable for predictive modeling and analysis

Predictive Modeling and Analysis Script
This Python script loads an Excel dataset, processes it to drop specified columns, allows the user to input a target variable, and then performs various predictive modeling tasks including hyperparameter tuning, evaluation, and feature importance analysis. It also generates visualizations for correlation matrices and SHAP values to interpret model predictions.

Prerequisites
Ensure you have the following Python libraries installed:

pandas
seaborn
matplotlib
scikit-learn
numpy
shap
openpyxl (for reading Excel files)
You can install the required libraries using pip:

bash
Copy code
pip install pandas seaborn matplotlib scikit-learn numpy shap openpyxl
Script Overview
The script performs the following steps:

Load the Excel file.
Drop specified columns.
Prompt for the target variable.
Generate and visualize a correlation matrix.
Prepare the data for the predictive model.
Split the data into training and testing sets.
Create and train a Random Forest Regressor model.
Perform hyperparameter tuning using GridSearchCV.
Evaluate the model using mean squared error and R-squared metrics.
Calculate cross-validation scores.
Plot actual vs predicted values.
Plot feature importances.
Generate SHAP values for model interpretation.
Usage
Update the file_path variable with the path to your Excel file.
Run the script.
When prompted, enter the name of the target variable from your dataset.
Detailed Function Description
Loading and Processing the Data
python
Copy code
file_path = 'C:/Users/dave2/OneDrive/Work/New Run/Flight and Inventory Report_excel data.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)
columns_to_omit = ["Date", "Owner Org Code", "TEC", "BUNO", "MC %", "FMC %"]
df = df.drop(columns=columns_to_omit)
target_variable = input("Please enter the target variable: ")
This section loads the Excel file and drops the specified columns. The user is then prompted to input the target variable for the predictive model.

Correlation Matrix
python
Copy code
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
Generates and visualizes the correlation matrix of the dataset.

Preparing the Data
python
Copy code
X = df.drop(columns=[target_variable])
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Splits the dataset into features (X) and target variable (y), and further splits it into training and testing sets.

Model Training and Hyperparameter Tuning
python
Copy code
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
Trains a Random Forest Regressor and performs hyperparameter tuning using GridSearchCV.

Model Evaluation
python
Copy code
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f'Cross-Validation R² Scores: {scores}')
Evaluates the model using mean squared error, R-squared metrics, and cross-validation scores.

Visualizations
Actual vs Predicted Values
python
Copy code
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"Actual vs Predicted: {target_variable}")
plt.show()
Feature Importances
python
Copy code
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
SHAP Values
python
Copy code
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
Generates and visualizes SHAP values for model interpretation.

Notes
Ensure the Excel file path is correct.
Input the correct target variable name when prompted.
