# ⭐ Star Type Classifier — ML & Deep Learning

An end-to-end Python application that classifies stars into 6 stellar types using classical machine learning and a TensorFlow neural network, all wrapped in an interactive **Tkinter GUI** with live training charts and real-time predictions.

---

## 📌 Project Overview

This project trains and compares multiple ML models on the [Star Type Classification dataset](https://www.kaggle.com/datasets/brsdincer/star-type-classification) from Kaggle. Users can select a model, tune hyperparameters, visualize live training metrics, inspect confusion matrices, and predict star types from custom input values — all within a dark-themed desktop GUI.

---

## 🌌 Star Types Classified

| Type ID | Star Type     |
|---------|---------------|
| 0       | Brown Dwarf   |
| 1       | Red Dwarf     |
| 2       | White Dwarf   |
| 3       | Main Sequence |
| 4       | Supergiant    |
| 5       | Hypergiant    |

---

## 📁 Project Structure

```
star-classification/
│
├── star_classification.py   # Main application (GUI + ML pipeline)
├── Stars.csv                # Dataset (240 star records)
└── README.md
```

---

## 🧪 Dataset

**File:** `Stars.csv`  
**Records:** 240 stars  
**Features:**

| Column           | Description                        |
|------------------|------------------------------------|
| `Temperature`    | Surface temperature (K)            |
| `L`              | Luminosity relative to the Sun     |
| `R`              | Radius relative to the Sun         |
| `A_M`            | Absolute magnitude                 |
| `Color`          | Star color (e.g., Red, Blue, White)|
| `Spectral_Class` | Spectral class (O, B, A, F, G, K, M) |
| `Type`           | Star type label (0–5)              |

---

## 🤖 Models

| Model                  | Library      |
|------------------------|--------------|
| Random Forest          | scikit-learn |
| Gradient Boosting      | scikit-learn |
| SVM (RBF Kernel)       | scikit-learn |
| Neural Network         | TensorFlow / Keras |

The neural network supports configurable hidden layer sizes, dropout rate, epochs, and batch size — all adjustable through the GUI sidebar.

---

## 🖥️ GUI Features

- **Live Training Charts** — accuracy and loss curves update epoch by epoch
- **Model Comparison Tab** — bar chart comparing test accuracy across all models
- **Confusion Matrix** — visual heatmap after each model trains
- **Data Explorer Tab** — scatter plots and distribution charts of the dataset
- **Custom Prediction Panel** — enter star physical properties and get an instant star type prediction with confidence score
- **5-Fold Cross-Validation** scores displayed per model

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/star-classification.git
cd star-classification
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

> Python 3.8+ recommended. Tkinter is included with most Python installations.

### 3. Run the app
```bash
python star_classification.py
```

> Make sure `Stars.csv` is in the **same directory** as `star_classification.py`.

---

## 🔬 Preprocessing Pipeline

1. **Categorical Encoding** — `Color` and `Spectral_Class` are label-encoded
2. **Log Transformation** — `Temperature`, `L`, and `R` are log-transformed to reduce skew
3. **Standard Scaling** — all features are scaled to zero mean and unit variance
4. **Train/Test Split** — 80/20 stratified split

---

## 📊 Sample Results

| Model             | Test Accuracy |
|-------------------|---------------|
| Random Forest     | ~98%          |
| Gradient Boosting | ~97%          |
| SVM (RBF)         | ~97%          |
| Neural Network    | ~97–99%       |

> Results may vary slightly across runs due to randomness in training.

---

## 🧰 Tech Stack

- **Python** — core language
- **Tkinter** — GUI framework
- **Pandas / NumPy** — data handling
- **Matplotlib** — charting and visualization
- **scikit-learn** — classical ML models
- **TensorFlow / Keras** — deep learning

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Made with ☕ and curiosity about the cosmos.
