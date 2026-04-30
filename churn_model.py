# ============================================
# CHURN PREDICTION PROJECT
# Goal: Predict which customers will leave
# ============================================

# These are our tools - think of them like apps we're opening
import pandas as pd          # pandas = works like Excel, handles tables of data
import numpy as np           # numpy = handles numbers and math operations
import matplotlib.pyplot as plt  # matplotlib = draws charts and graphs
import seaborn as sns        # seaborn = makes prettier charts than matplotlib
import warnings
warnings.filterwarnings('ignore')  # hides annoying warning messages while we work

# sklearn is our machine learning library - we'll use different parts of it
from sklearn.model_selection import train_test_split      # splits data into train/test
from sklearn.ensemble import RandomForestClassifier       # our prediction model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # measures how good our model is
from sklearn.preprocessing import LabelEncoder           # converts text like "Yes/No" to numbers

print("✅ Step 1 done - all tools loaded!")

# ── STEP 2: Load the data ──────────────────────────────────────
# We're using a real telecom dataset from IBM
# It has 7,043 customers with 21 features like:
# monthly charges, contract type, internet service, etc.
# The last column "Churn" tells us if they left (Yes) or stayed (No)

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# pd.read_csv() reads a CSV file and turns it into a table (called a DataFrame)
# think of it like opening an Excel file in Python
df = pd.read_csv(url)

print(f"✅ Step 2 done - data loaded successfully!")

# ── STEP 3: First look at the data ────────────────────────────
# A good engineer NEVER touches data without looking at it first
# This is called EDA - Exploratory Data Analysis

# .shape tells us (number of rows, number of columns)
print(f"\n📊 Dataset size: {df.shape[0]} customers, {df.shape[1]} columns")

# .head() shows the first 5 rows - like peeking at the top of a spreadsheet
print(f"\n📋 First 5 customers:")
print(df.head())

# checking what columns we have - these are our "features" (inputs to the model)
print(f"\n🗂  All columns available:")
print(list(df.columns))

# most important check - how many customers churned vs stayed?
# if 95% stayed and 5% churned, our model needs special handling (class imbalance)
print(f"\n🔍 How many customers churned?")
print(df['Churn'].value_counts())
print(f"\nChurn rate: {round(df['Churn'].value_counts(normalize=True)['Yes'] * 100, 1)}%")

# .info() shows us column types and whether any values are missing
# missing values = a problem we'll fix in the next step
print(f"\n🔬 Data types and missing values:")
print(df.info())
# ── STEP 3: Clean the data ─────────────────────────────────────
# Raw data is always messy. This is the most important step.
# Garbage in = garbage out. If we feed bad data to the model,
# we'll get bad predictions. Simple as that.

print("\n🧹 Starting data cleaning...")

# PROBLEM 1: TotalCharges is text but should be a number
# Why? Because someone entered blank spaces " " instead of 0
# pd.to_numeric() converts it to a number, and errors='coerce'
# means "if you can't convert it, just put NaN (empty) instead of crashing"
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Now let's see how many empty values that created
print(f"Empty values in TotalCharges: {df['TotalCharges'].isnull().sum()}")

# PROBLEM 2: Fill those empty values with 0
# Why 0? Because if TotalCharges is empty, the customer is likely brand new
# and hasn't been charged yet - so 0 makes logical sense
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# PROBLEM 3: Drop the customerID column
# Why? Because customerID is just a random ID like "7590-VHVEG"
# It has zero relationship to whether someone churns or not
# Keeping it would confuse the model
df = df.drop('customerID', axis=1)

# PROBLEM 4: Convert Yes/No text columns to numbers (1 and 0)
# Why? Because ML models only understand numbers, not words
# Yes = 1, No = 0
# We find all columns that only contain "Yes" or "No" and convert them
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                  'Churn', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0,
                           'No phone service': 0,
                           'No internet service': 0})

# PROBLEM 5: Convert gender to a number
# Male = 1, Female = 0 (this choice is arbitrary, just needs to be a number)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# PROBLEM 6: Convert remaining text columns using one-hot encoding
# These are columns with more than 2 options like:
# InternetService = "DSL", "Fiber optic", "No"
# Contract = "Month-to-month", "One year", "Two year"
# PaymentMethod = 4 different options
# One-hot encoding creates separate columns for each option
# Example: Contract becomes Contract_OneYear, Contract_TwoYear, Contract_Monthtomonth
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])

print("✅ Step 3 done - data cleaned!")
print(f"\n📊 Dataset shape after cleaning: {df.shape}")
print(f"\n🔢 All columns are now numbers:")
print(df.dtypes)
print(f"\n👀 Sample of cleaned data:")
print(df.head(3))
# ── STEP 4: Train the model ────────────────────────────────────
print("\n🤖 Starting model training...")

# First we separate our data into:
# X = the inputs (everything we know about the customer)
# y = the answer (did they actually churn? 1=yes, 0=no)
# Think of X as the question and y as the answer key
X = df.drop('Churn', axis=1)   # everything except the Churn column
y = df['Churn']                 # only the Churn column

print(f"Input features (X): {X.shape[1]} columns")
print(f"Target (y): {y.shape[0]} customers")

# Now we split into training data and testing data
# 80% of customers go to training (model learns from these)
# 20% go to testing (we hide these from the model, then check its predictions)
# random_state=42 just means "split the same way every time we run this"
# so our results are reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📚 Training set: {X_train.shape[0]} customers (model learns from these)")
print(f"🧪 Testing set:  {X_test.shape[0]} customers (model never sees these until prediction)")

# Random Forest is like asking 100 different decision trees the same question
# and going with the majority vote
# It's one of the most reliable models for this kind of problem
# n_estimators=100 means we're building 100 trees
# random_state=42 means reproducible results every time
model = RandomForestClassifier(n_estimators=100, random_state=42)

# .fit() is where the actual learning happens
# we're showing the model the training customers and their churn answers
print("\n⏳ Training in progress...")
model.fit(X_train, y_train)
print("✅ Model trained!")

# Now we ask the model to predict churn for the test customers
# remember - it has NEVER seen these customers before
predictions = model.predict(X_test)

# ── STEP 5: Measure how good our model is ─────────────────────
# Accuracy = out of all predictions, what % were correct?
accuracy = accuracy_score(y_test, predictions)
print(f"\n🎯 Model Accuracy: {round(accuracy * 100, 2)}%")

# Classification report gives us deeper insight:
# Precision = when model says "will churn", how often is it right?
# Recall = out of all customers who actually churned, how many did we catch?
# F1 Score = balance between precision and recall
print("\n📊 Detailed Performance Report:")
print(classification_report(y_test, predictions, target_names=['Stayed', 'Churned']))

# Confusion matrix shows us:
# True Positives  = correctly predicted churn
# True Negatives  = correctly predicted stayed
# False Positives = said they'd churn but they stayed (false alarm)
# False Negatives = said they'd stay but they churned (missed!)
print("🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)
print("\nReading the matrix:")
print(f"  Correctly predicted STAYED:  {cm[0][0]}")
print(f"  Correctly predicted CHURNED: {cm[1][1]}")
print(f"  False alarms (said churn, actually stayed): {cm[0][1]}")
print(f"  Missed churners (said stayed, actually churned): {cm[1][0]}")
# ── STEP 6: Visualize results ──────────────────────────────────
# Charts tell the story better than numbers alone
# These are the kind of visuals you'd present to a business team

print("\n📈 Generating charts...")

# We'll create 3 charts in one figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Churn Prediction Model - Results', fontsize=16, fontweight='bold')

# ── Chart 1: Confusion Matrix Heatmap ─────────────────────────
# A heatmap makes the confusion matrix much easier to read
# Dark color = high number, light color = low number
sns.heatmap(
    cm,
    annot=True,           # show the actual numbers inside each box
    fmt='d',              # format as integers (not decimals)
    cmap='Blues',         # use blue color scale
    xticklabels=['Predicted Stayed', 'Predicted Churned'],
    yticklabels=['Actually Stayed', 'Actually Churned'],
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=11)
axes[0].set_xlabel('Predicted', fontsize=11)

# ── Chart 2: Feature Importance ───────────────────────────────
# This answers: which customer details matter most for predicting churn?
# Random Forest can tell us this automatically
# High importance = that feature has a big influence on the prediction
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})

# sort by importance and take top 10 most influential features
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

sns.barplot(
    data=feature_importance,
    x='importance',
    y='feature',
    palette='viridis',
    ax=axes[1]
)
axes[1].set_title('Top 10 Most Important Features', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Importance Score', fontsize=11)
axes[1].set_ylabel('Feature', fontsize=11)

# ── Chart 3: Churn Distribution ───────────────────────────────
# Simple bar chart showing the original churn split in our dataset
# This reminds the viewer WHY this problem matters
churn_counts = df['Churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']   # green for stayed, red for churned

axes[2].bar(
    ['Stayed', 'Churned'],
    churn_counts.values,
    color=colors,
    edgecolor='white',
    linewidth=1.5
)
axes[2].set_title('Churn Distribution in Dataset', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Number of Customers', fontsize=11)

# add the actual numbers on top of each bar
for i, count in enumerate(churn_counts.values):
    axes[2].text(i, count + 30, str(count), ha='center', fontsize=12, fontweight='bold')

# ── Save and show ──────────────────────────────────────────────
plt.tight_layout()

# saves the chart as an image file in your project folder
# this is what you'll upload to GitHub!
plt.savefig('churn_results.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as churn_results.png in your project folder!")

plt.show()
print("\n🚀 Project complete! You just built a real ML churn prediction model.")
print(f"   Final accuracy: {round(accuracy * 100, 2)}%")
print(f"   Customers analyzed: {len(df)}")
print(f"   Features used: {X.shape[1]}")