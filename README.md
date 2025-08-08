# KNN Classifier (Iris Dataset)

## 📌 Project Overview
This project implements a **K-Nearest Neighbors (KNN) classifier** on the classic **Iris dataset** using Python and scikit-learn.  
It includes:
- Data loading and preprocessing
- Feature scaling
- Cross-validation to select the optimal value of **k**
- Model training and evaluation
- Confusion matrix visualization
- Saving the trained model and scaler for later use

---

## 🛠 Tech Stack
- **Python 3.8+**
- **pandas** — Data manipulation
- **scikit-learn** — ML algorithms & metrics
- **matplotlib** / **seaborn** — Visualization
- **joblib** — Saving model artifacts

---

## 📂 Project Structure

```
knn/
│── knn_classifier.py # Main script
│── knn_k_selection.png # CV accuracy vs k plot
│── knn_confusion_matrix.png # Confusion matrix heatmap
│── knn_model.joblib # Saved trained model
│── knn_scaler.joblib # Saved scaler for preprocessing
│── README.md # This file

```

---

## 4. Setup & Installation

### Clone Repository
```bash
git clone https://github.com/<green1210>/<your-repo>.git
cd <your-repo>/knn
```

## 5. Create Virtual Environment (Recommended)
---
### Windows (PowerShell)
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### Linux/macOS
python3 -m venv venv
source venv/bin/activate

### Install Dependencies
pip install -r requirements.txt

### If no requirements.txt
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```
---

## 6. How to Run

```
python knn_classifier.py
```

---

## 7. Outputs & Screenshots

### Best k Selection Plot
Shows cross-validation accuracy for different k values.

### Confusion Matrix
Visualizes classification performance per class.

### Terminal Output
Example run showing accuracy, classification report, and chosen k.

---

## 8. Results
Achieved ~97% accuracy on the Iris test set (may vary per run).

Optimal k selected via 5-fold cross-validation.

---

## 9. License
This project is licensed under the MIT License — feel free to use, modify, and share.


---

