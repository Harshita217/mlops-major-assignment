import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import os

def main():
    model_path = "models/savedmodel.pth"
    test_path = "models/test_data.npz"

    if not os.path.exists(model_path) or not os.path.exists(test_path):
        print("Model or test data not found. Run train.py first.")
        return

    clf = joblib.load(model_path)
    data = np.load(test_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
