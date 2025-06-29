
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === CONFIG ===
lambda_ = 0.005
cv_folds = 5
log_file = "xgb_results.txt"

# === LOAD DATA ===
df = pd.read_csv("EDITED NEWS.csv")
df.columns = df.columns.str.strip()
df.rename(columns={"Movement": "% Movement"}, inplace=True)
df["% Movement"] = pd.to_numeric(df["% Movement"], errors='coerce')
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["DaysSinceEvent"] = (pd.Timestamp.today() - df["Date"]).dt.days
df = df[df["DaysSinceEvent"] >= 0]

# === Movement Class ===
def classify_movement(x):
    if x > 2:
        return "Up"
    elif x < -2:
        return "Down"
    else:
        return "Neutral"

df["Movement Class"] = df["% Movement"].apply(classify_movement)
df.dropna(subset=["Sector", "Event Type", "Movement Class"], inplace=True)

# === Apply Decay ===
df["DecayWeight"] = np.exp(-lambda_ * df["DaysSinceEvent"])

# === Features and Encoding ===
X = df[["Sector", "Event Type", "DaysSinceEvent"]]
y = df["Movement Class"]
sample_weights = df["DecayWeight"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y_encoded, sample_weights, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Preprocessor ===
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ["Sector", "Event Type"])
], remainder='passthrough')

# === XGBoost Model with Tuned Parameters ===
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# === Train Model ===
model.fit(X_train, y_train, classifier__sample_weight=weights_train)

# === Predict and Evaluate ===
y_pred = model.predict(X_test)
decoded_preds = label_encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# === Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d",
            cmap="Blues", xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Weighted Decay + Tuned Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# === Cross-Validation ===
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')

# === Recommendation System ===
def get_recommendation(label):
    if label == "Up":
        return "Add"
    elif label == "Down":
        return "Reduce"
    else:
        return "Hold"

results_df = X_test.copy()
results_df["Predicted Class"] = decoded_preds
results_df["Recommendation"] = results_df["Predicted Class"].apply(get_recommendation)

# === Save Model and Label Encoder ===
joblib.dump(model, "xgb_model_with_decay_tuned.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# === Write to Log File ===
with open(log_file, "w") as f:
    f.write("=== Model Evaluation ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix) + "\n")
    f.write(f"Cross-Validation Scores: {cv_scores}\n")
    f.write(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}\n")
    f.write("\nSample Predictions:\n")
    f.write(str(results_df[["Sector", "Event Type", "DaysSinceEvent", "Predicted Class", "Recommendation"]].head()))
