
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pandas as pd

# 1. Create a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# 2. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Test different numbers of estimators
n_estimators_values = [5, 10, 50, 100, 200, 500]
results = []

for n in n_estimators_values:
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")

    # Store results
    results.append({
        "Number of estimators": n,
        "Test accuracy (%)": round(accuracy * 100, 2),
        "F1-score": round(f1, 4),
        "F1-score (%)": round(f1 * 100, 2)
    })

# 4. Display the results table
results_table = pd.DataFrame(results)
print(results_table)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

for depth in [1, 2, 3, 5]:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train, model.predict(X_train)
    )
    test_acc = accuracy_score(
        y_test, model.predict(X_test)
    )

    print(
        f"max_depth={depth}: "
        f"Train Accuracy={train_acc:.3f}, "
        f"Test Accuracy={test_acc:.3f}"
    )