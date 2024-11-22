#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr


# In[3]:


# Generate Synthetic Data
np.random.seed(42)
n_samples = 1000
data = {
 "Age": np.random.randint(13, 19, n_samples),
 "Gender": np.random.choice(["Male", "Female"], n_samples),
 "Ethnicity": np.random.choice(["Group A", "Group B", "Group C"], n_samples),
 "Parental_Education": np.random.randint(1, 5, n_samples),
 "Parental_Support": np.random.randint(1, 10, n_samples),
 "Weekly_Study_Hours": np.random.randint(1, 20, n_samples),
 "Absences": np.random.randint(0, 15, n_samples),
 "Sports": np.random.choice([0, 1], n_samples),
 "Music": np.random.choice([0, 1], n_samples),
 "Volunteering": np.random.choice([0, 1], n_samples),
 "Past_GPA": np.round(np.random.uniform(2.0, 4.0, n_samples), 2)
}


# In[4]:


# Generate Target Variables
data["Current_GPA"] = (
 0.3 * data["Weekly_Study_Hours"] -
 0.1 * data["Absences"] +
 0.2 * data["Parental_Support"] +
 0.15 * data["Past_GPA"] +
 0.1 * (data["Sports"] + data["Music"] + data["Volunteering"]) +
 np.random.normal(0, 0.3, n_samples)
)


# In[6]:


data["Current_GPA"] = np.clip(data["Current_GPA"], 0, 4)
data["At_Risk"] = (data["Current_GPA"] < 2.5).astype(int)


# In[8]:


df = pd.DataFrame(data)
# Convert Categorical Columns
df = pd.get_dummies(df, columns=["Gender", "Ethnicity"], drop_first=True)


# In[9]:


# Split Data
X = df.drop(columns=["Current_GPA", "At_Risk"])
y_gpa = df["Current_GPA"]
y_risk = df["At_Risk"]
X_train, X_test, y_train_gpa, y_test_gpa = train_test_split(X, y_gpa, test_size=0.2, random_state=42)
_, _, y_train_risk, y_test_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)


# In[10]:


# Model 1: Predicting GPA (Regression)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_gpa)
y_pred_gpa = reg_model.predict(X_test)
print("Regression - Mean Squared Error:", mean_squared_error(y_test_gpa, y_pred_gpa))


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your preprocessed DataFrame
X = df.drop(columns=["Current_GPA", "At_Risk"])
y = df["Current_GPA"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predictions
y_pred = reg_model.predict(X_test)

# Visualize Predicted vs Actual Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([0, 4], [0, 4], color='red', linestyle='--')  # Reference line
plt.title('Predicted vs Actual GPA')
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.grid()
plt.show()


# In[12]:


# Model 2: Identifying At-Risk Students (Classification)
clf_model = LogisticRegression()
clf_model.fit(X_train, y_train_risk)
y_pred_risk = clf_model.predict(X_test)
print("Classification - Accuracy Score:", accuracy_score(y_test_risk, y_pred_risk))
print("Confusion Matrix:\n", confusion_matrix(y_test_risk, y_pred_risk))


# In[17]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming df is your preprocessed DataFrame
X = df.drop(columns=["Current_GPA", "At_Risk"])
y = df["At_Risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf_model = LogisticRegression()
clf_model.fit(X_train, y_train)

# Predictions
y_pred = clf_model.predict(X_test)

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[13]:


# Analysis: Correlation Between Extracurriculars and GPA
extracurriculars = ["Sports", "Music", "Volunteering"]
for activity in extracurriculars:
 correlation, _ = pearsonr(df[activity], df["Current_GPA"])
 print(f"Correlation between {activity} and GPA: {correlation:.2f}")


# In[19]:


# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Features')
plt.show()


# In[18]:


# Feature importance for Linear Regression
feature_importance = reg_model.coef_

plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=feature_importance, palette='viridis')
plt.title('Feature Importance in Predicting GPA')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.xticks(rotation=45)
plt.show()


# In[20]:


# Pair plot for selected features
selected_features = ['Weekly_Study_Hours', 'Absences', 'Parental_Support', 'Past_GPA', 'Current_GPA']
sns.pairplot(df[selected_features], diag_kind='kde', palette='husl')
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()


# In[21]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='Sports', y='Current_GPA', data=df, palette='Set3')
plt.title('Impact of Sports Participation on GPA')
plt.xlabel('Sports Participation')
plt.ylabel('Current GPA')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Music', y='Current_GPA', data=df, palette='Set3')
plt.title('Impact of Music Participation on GPA')
plt.xlabel('Music Participation')
plt.ylabel('Current GPA')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Volunteering', y='Current_GPA', data=df, palette='Set3')
plt.title('Impact of Volunteering on GPA')
plt.xlabel('Volunteering')
plt.ylabel('Current GPA')
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr

# Regression Performance Summary
regression_mse = mean_squared_error(y_test_gpa, y_pred_gpa)

# Classification Performance Summary
classification_accuracy = accuracy_score(y_test_risk, y_pred_risk)
conf_matrix = confusion_matrix(y_test_risk, y_pred_risk)

# Correlation Analysis Summary
correlations = {
    activity: pearsonr(df[activity], df["Current_GPA"])[0]
    for activity in ["Sports", "Music", "Volunteering"]
}

# Create a Summary Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Regression: Predicted vs Actual
sns.scatterplot(x=y_test_gpa, y=y_pred_gpa, ax=axes[0, 0], color='blue', alpha=0.6)
axes[0, 0].plot([0, 4], [0, 4], color='red', linestyle='--')
axes[0, 0].set_title("Regression: Predicted vs Actual GPA")
axes[0, 0].set_xlabel("Actual GPA")
axes[0, 0].set_ylabel("Predicted GPA")

# 2. Classification: Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title(f"Classification Confusion Matrix\nAccuracy: {classification_accuracy:.2f}")
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("Actual")

# 3. Feature Correlation: Extracurriculars
sns.barplot(x=list(correlations.keys()), y=list(correlations.values()), palette="muted", ax=axes[1, 0])
axes[1, 0].set_title("Correlation Between Extracurriculars and GPA")
axes[1, 0].set_xlabel("Activity")
axes[1, 0].set_ylabel("Correlation Coefficient")

# 4. Overall Insights: Text Summary
axes[1, 1].axis('off')
text = (f"Regression Mean Squared Error: {regression_mse:.2f}\n"
        f"Classification Accuracy: {classification_accuracy:.2f}\n\n"
        f"Insights:\n"
        f"- Strong correlation observed between GPA and study hours.\n"
        f"- Absences negatively impact GPA.\n"
        f"- Extracurriculars (Sports, Music, Volunteering) show moderate positive correlation with GPA.")
axes[1, 1].text(0.5, 0.5, text, fontsize=12, ha='center', va='center', wrap=True)

# Adjust layout and show the summary
plt.tight_layout()
plt.suptitle("Summary: Academic Achievement Predictive Analysis", y=1.02, fontsize=16)
plt.show()

