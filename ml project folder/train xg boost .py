import pandas as pd
import joblib
from sklearn.calibration import LabelEncoder
df=pd.read_csv(r"C:\Users\abhis\OneDrive\Desktop\project folder\EDITED NEWS .csv")
print(df)
df.drop_duplicates()
df.info()
df.describe()
df.head()
df.drop_duplicates(inplace=True)
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\abhis\OneDrive\Desktop\project folder\EDITED NEWS .csv")
df.columns = df.columns.str.strip()  # remove leading/trailing spaces

# Rename and convert movement column
df.rename(columns={"Movement": "% Movement"}, inplace=True)
df["% Movement"] = pd.to_numeric(df["% Movement"], errors='coerce')

# Convert date
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["DaysSinceEvent"] = (pd.Timestamp.today() - df["Date"]).dt.days
# Remove rows where event date is in the future (negative days)
df = df[df["DaysSinceEvent"] >= 0]


# Now create Movement Class
def classify_movement(x):
    if x > 2:
        return "Up"
    elif x < -2:
        return "Down"
    else:
        return "Neutral"

df["Movement Class"] = df["% Movement"].apply(classify_movement)

df.dropna(subset=["Sector", "Event Type", "Movement Class"], inplace=True)

# STEP 6: Define features and target
X = df[["Sector", "Event Type", "DaysSinceEvent"]]
y = df["Movement Class"]

# STEP 7: Encode target (Movement Class) into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Print label mapping (optional but useful)
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Now  for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: One-hot encode categorical columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ["Sector", "Event Type"])
], remainder='passthrough')  # keeps DaysSinceEvent

# STEP 2: Build pipeline with XGBoost
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# STEP 3: Train the model
model.fit(X_train, y_train)

# STEP 4: Predict
y_pred = model.predict(X_test)

# STEP 5: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# STEP 6: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# Use encoded y (y_encoded) from earlier steps

#  Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {scores}")
print(f"Mean Accuracy: {np.mean(scores):.4f}")
# Predict on the test set
y_pred = model.predict(X_test)

# Decode predictions back to class labels
decoded_preds = label_encoder.inverse_transform(y_pred)

# Add recommendation logic
def get_recommendation(label):
    if label == "Up":
        return "Add"
    elif label == "Down":
        return "Reduce"
    else:
        return "Hold"

# Create DataFrame of predictions + recommendations
results_df = X_test.copy()
results_df["Predicted Class"] = decoded_preds
results_df["Recommendation"] = results_df["Predicted Class"].apply(get_recommendation)

# Show final predictions and recommendations
print("\n Final Prediction Results with Recommendations:")
print(results_df[["Sector", "Event Type", "DaysSinceEvent", "Predicted Class", "Recommendation"]].head())


joblib.dump(model, "xgb_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")