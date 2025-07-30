# lung_cancer_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib


# Load dataset
df = pd.read_csv("/Users/shrinithask/Downloads/dataset.csv")

# Encode categorical variables
le_gender = LabelEncoder()
le_cancer = LabelEncoder()

df['GENDER'] = le_gender.fit_transform(df['GENDER'])         # M/F -> 1/0
df['LUNG_CANCER'] = le_cancer.fit_transform(df['LUNG_CANCER'])  # YES/NO -> 1/0

# Features and target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model for Streamlit app
joblib.dump(model, "lung_cancer_model.pkl")
