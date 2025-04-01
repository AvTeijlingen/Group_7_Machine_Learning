from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from load_data import load_data
from sklearn.preprocessing import LabelEncoder


# === Load Your Multidimensional Dataset ===
df = load_data()  # Load real dataset
X = df.drop(columns=["label"])  # Drop ID, keep features
y = df["label"]  # Target variable

# Convert 'GIST' and 'non-GIST' to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # This will convert 'GIST' to 1, 'non-GIST' to 0

# Hyperparameter grids
param_grid_logreg = {'C': [0.1, 1, 10], 'penalty': ['l2']}
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# 5-fold cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Classifiers and hyperparameter grids
classifiers = {
    'Logistic Regression': (LogisticRegression(), param_grid_logreg),
    'Random Forest': (RandomForestClassifier(), param_grid_rf),
    'SVM': (SVC(probability=True), param_grid_svm)  # Probability=True for ROC-AUC
}

# Store results per model
results_per_model = {name: [] for name in classifiers.keys()}
best_hyperparameters = {}

# === Start cross-validation ===
for fold, (train_index, test_index) in enumerate(skf.split(X, y_encoded), start=1):
    print(f"\n=== Fold {fold} ===")

    # Split data into 80% training & 20% test
    X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_full, y_test = y_encoded[train_index], y_encoded[test_index]

    # Split the training set again into 80% training & 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    for classifier_name, (clf, param_grid) in classifiers.items():
        print(f"\n🔹 Training {classifier_name} on fold {fold}")

        # Hyperparameter optimization with GridSearch
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Select the best model
        best_model = grid_search.best_estimator_

        # Store best hyperparameters
        best_hyperparameters.setdefault(classifier_name, []).append(grid_search.best_params_)

        # TEST PHASE: Test the model on the original test set from the fold
        y_preds = best_model.predict(X_test)
        y_probs = best_model.predict_proba(X_test)[:, 1]  # For ROC-AUC

        # Calculate evaluation metrics
        metrics = {
            "Fold": fold,
            "Precision": precision_score(y_test, y_preds),
            "Recall": recall_score(y_test, y_preds),
            "F1-score": f1_score(y_test, y_preds),
            "ROC-AUC": roc_auc_score(y_test, y_probs)
        }

        # Print test results for this fold
        print(f" Test results for {classifier_name} on Fold {fold}:")
        for metric, value in metrics.items():
            if metric != "Fold":
                print(f"    {metric}: {value:.4f}")

        # Save results
        results_per_model[classifier_name].append(metrics)

# === Print summary of results per model ===
print("\n=== INDIVIDUAL MODEL RESULTS ===")
for classifier, results in results_per_model.items():
    df_results = pd.DataFrame(results)
    
    print(f"\n🔹 {classifier}:")
    print(df_results.to_string(index=False))

    # Display best hyperparameters per fold
    print("\n Best hyperparameters per fold:")
    for i, params in enumerate(best_hyperparameters[classifier], start=1):
        print(f"  Fold {i}: {params}")

    # Optionally, you can calculate the average metrics across folds
    avg_metrics = df_results.mean()
    print(f"\n Average metrics for {classifier}:")
    print(avg_metrics)