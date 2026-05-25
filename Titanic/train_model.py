import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import load_model
import joblib

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Titanic-Dataset.csv")

# =========================
# SELECT FEATURES
# =========================

X = df[['Pclass', 'Age', 'Fare']].copy()

# Fill missing Age values
X['Age'] = X['Age'].fillna(X['Age'].mean())

y = df['Survived']

# =========================
# NORMALIZATION
# =========================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# BUILD ANN MODEL
# =========================

model = Sequential([
    Input(shape=(3,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10,
    verbose=1
)

# =========================
# SAVE MODEL
# =========================

model.save("model/titanic_ann_model.h5")

print("Model Saved Successfully!")