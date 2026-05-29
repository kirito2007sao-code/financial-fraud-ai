import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ======================
# LOAD DATA
# ======================

df = pd.read_csv("data/financial_statement.csv")

print(df.head())

# ======================
# TARGET COLUMN
# ======================

target_column = "Financial_Status"

# ======================
# FEATURES & LABEL
# ======================

X = df.drop(target_column, axis=1)
y = df[target_column]

# ======================
# CHUYỂN LABEL THÀNH SỐ
# ======================

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# ======================
# TRAIN TEST SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================
# MODEL
# ======================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# ======================
# TRAIN
# ======================

model.fit(X_train, y_train)

# ======================
# PREDICT
# ======================

predictions = model.predict(X_test)

# ======================
# ACCURACY
# ======================

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.2f}")

# ======================
# CREATE MODEL FOLDER
# ======================

os.makedirs("model", exist_ok=True)

# ======================
# SAVE MODEL
# ======================

joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(X.columns.tolist(), "model/model_columns.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("\nModel saved successfully!")