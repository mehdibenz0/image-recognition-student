import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open("dataset.pkl", "rb") as f:
    X, y = pickle.load(f)

X_flat = X.reshape(len(X), -1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# TODO 1: Choose the ratio of the training and test sets (use 20% for test)
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_encoded,
    test_size=,
    random_state=42
)

# TODO 2: Create and train a logistic regression model
model = LogisticRegression(max_iter=1000)


# TODO 3: Predict on the test set
y_pred = 

acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc:.2f}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("labels.pkl", "wb") as f:
    pickle.dump(le, f)
