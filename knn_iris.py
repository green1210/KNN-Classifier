
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_data(path='iris.csv'):
    df = pd.read_csv(path)
    return df

def explore_data(df):
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nClass distribution:")
    print(df['species'].value_counts())

    sns.pairplot(df, hue='species', corner=True)
    plt.suptitle('Iris Dataset Pairplot', y=1.02)
    plt.show()

def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test),
                cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def tune_hyperparameters(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    print(f"\nBest k: {grid.best_params_['n_neighbors']}")
    print(f"Best CV score: {grid.best_score_:.4f}")

    results = pd.DataFrame(grid.cv_results_)
    plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o')
    plt.xlabel('k (n_neighbors)')
    plt.ylabel('CV Accuracy')
    plt.title('K vs Cross-Validation Accuracy')
    plt.grid(True)
    plt.show()

    return grid.best_estimator_

def save_model(model, scaler, model_path='knn_model.joblib', scaler_path='scaler.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to '{model_path}'")
    print(f"Scaler saved to '{scaler_path}'")

def main():
    
    df = load_data('iris.csv')

    explore_data(df)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    knn_model = train_knn(X_train, y_train, n_neighbors=5)

    evaluate_model(knn_model, X_test, y_test)

    best_model = tune_hyperparameters(X_train, y_train)

    print("\nEvaluating tuned model:")
    evaluate_model(best_model, X_test, y_test)

    save_model(best_model, scaler)

if __name__ == "__main__":
    main()