import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/spotify_dataset.csv")

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. CREATE POPULARITY CATEGORY
# ============================================================

def popularity_category(popularity):
    if popularity <= 39:
        return "Not Popular"
    elif popularity <= 69:
        return "Moderately Popular"
    else:
        return "Popular"


df["popularity_category"] = df["popularity"].apply(
    popularity_category
)

print("\nPopularity category distribution:")
print(df["popularity_category"].value_counts())

print("\nPopularity category percentage:")
print(
    (df["popularity_category"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())

columns_to_remove = [
    "Unnamed: 0",
    "track_id",
    "artists",
    "album_name",
    "track_name"
]

df = df.drop(columns=columns_to_remove)

print("\nUnnecessary columns removed successfully.")


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

X = df[features]
y = df["popularity_category"]

print("\nFeatures used for the ML model:")
print(features)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

os.makedirs("plots", exist_ok=True)

# 5.1 Popularity distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="popularity_category"
)

plt.title("Spotify Song Popularity Category Distribution")
plt.xlabel("Popularity Category")
plt.ylabel("Number of Songs")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("plots/popularity_distribution.png")
plt.close()


# 5.2 Correlation heatmap
plt.figure(figsize=(10, 7))

correlation = X.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Audio Features")

plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()


# 5.3 Audio feature distributions
X.hist(
    figsize=(15, 12),
    bins=30
)

plt.suptitle(
    "Distribution of Spotify Audio Features",
    fontsize=16
)

plt.tight_layout()
plt.savefig("plots/audio_feature_distributions.png")
plt.close()

print("\nEDA graphs saved successfully!")


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed successfully!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 7. FEATURE SCALING FOR LOGISTIC REGRESSION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed successfully!")


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

y_pred_logistic = logistic_model.predict(
    X_test_scaled
)

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

print("\nLogistic Regression training completed successfully!")
print(f"Accuracy: {logistic_accuracy:.6f}")

print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_logistic
    )
)


# ============================================================
# 9. DECISION TREE
# ============================================================

decision_tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(
    X_train,
    y_train
)

y_pred_tree = decision_tree_model.predict(
    X_test
)

tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

print("\nDecision Tree training completed successfully!")
print(f"Accuracy: {tree_accuracy:.6f}")

print("\nDecision Tree Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_tree
    )
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train,
    y_train
)

y_pred_forest = random_forest_model.predict(
    X_test
)

forest_accuracy = accuracy_score(
    y_test,
    y_pred_forest
)

print("\nRandom Forest training completed successfully!")
print(f"Accuracy: {forest_accuracy:.6f}")

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_forest
    )
)


# ============================================================
# 11. CONFUSION MATRICES
# ============================================================

models_predictions = [
    (
        "Logistic Regression",
        logistic_model.classes_,
        y_pred_logistic,
        "plots/logistic_confusion_matrix.png"
    ),
    (
        "Decision Tree",
        decision_tree_model.classes_,
        y_pred_tree,
        "plots/decision_tree_confusion_matrix.png"
    ),
    (
        "Random Forest",
        random_forest_model.classes_,
        y_pred_forest,
        "plots/random_forest_confusion_matrix.png"
    )
]

for model_name, classes, predictions, file_path in models_predictions:

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    display.plot()

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

print("\nConfusion matrices saved successfully!")


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_logistic),
        accuracy_score(y_test, y_pred_tree),
        accuracy_score(y_test, y_pred_forest)
    ],
    "Precision": [
        precision_score(
            y_test,
            y_pred_logistic,
            average="weighted"
        ),
        precision_score(
            y_test,
            y_pred_tree,
            average="weighted"
        ),
        precision_score(
            y_test,
            y_pred_forest,
            average="weighted"
        )
    ],
    "Recall": [
        recall_score(
            y_test,
            y_pred_logistic,
            average="weighted"
        ),
        recall_score(
            y_test,
            y_pred_tree,
            average="weighted"
        ),
        recall_score(
            y_test,
            y_pred_forest,
            average="weighted"
        )
    ],
    "F1-Score": [
        f1_score(
            y_test,
            y_pred_logistic,
            average="weighted"
        ),
        f1_score(
            y_test,
            y_pred_tree,
            average="weighted"
        ),
        f1_score(
            y_test,
            y_pred_forest,
            average="weighted"
        )
    ]
})

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 13. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# Feature importance plot
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Audio Feature")

plt.tight_layout()
plt.savefig(
    "plots/random_forest_feature_importance.png"
)
plt.close()

print("\nFeature importance plot saved successfully!")


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model_package = {
    "model": random_forest_model,
    "features": features,
    "classes": random_forest_model.classes_
}

joblib.dump(
    model_package,
    "models/best_model.pkl"
)

print("\n" + "=" * 60)
print("BEST MODEL SAVED SUCCESSFULLY")
print("=" * 60)
print("Model: Random Forest Classifier")
print("File: models/best_model.pkl")

print("\nTraining pipeline completed successfully!")