# Digit Recognizer — From-Scratch Neural Network (NumPy)

A minimal, from-scratch neural network for handwritten digit classification on the classic [Kaggle Digit Recognizer (MNIST)](https://www.kaggle.com/competitions/digit-recognizer) dataset.  
No PyTorch, no TensorFlow — just **NumPy**, a **single hidden layer**, **ReLU + Softmax**, and manual **backprop with gradient descent**.

> Based on the notebook: [`digit-recognizer`](https://www.kaggle.com/code/ihavealaptop/digit-recognizer)

---

## ✨ What’s inside

- Loads Kaggle’s `train.csv` and `test.csv`
- Normalizes pixel values to `[0,1]`
- Splits a **dev/validation set** from the training data
- Builds a tiny MLP: **784 → 10 (ReLU) → 10 (Softmax)**
- Implements **forward pass, one-hot encoding, backprop, and parameter updates** by hand
- Trains with simple **gradient descent**
- Includes a small helper to visualize random predictions (optional)

---

## 🗂️ Repository structure

```
.
├─ README.md
└─ digit-recognizer.ipynb     # This notebook
```

> If you prefer a script, I can convert the notebook to a clean `.py` module with a CLI.

---

## 📊 Dataset

- **Source:** Kaggle Digit Recognizer (MNIST)
- **Files:**
  - `train.csv` — 42,000 rows, first column is `label`, remaining 784 columns are pixel values.
  - `test.csv` — 28,000 rows, **no labels** (for Kaggle submissions).

> Because `test.csv` has no labels, you cannot compute accuracy on it. Use a **held-out split from `train.csv`** for evaluation (the notebook already creates a `dev` split).

---

## 🔧 Setup

### Option A: Run on Kaggle
1. Open this notebook in Kaggle Notebooks.
2. In **Add data**, attach the **Digit Recognizer** competition dataset.
3. Run all cells.

### Option B: Run locally

```bash
# 1) Create & activate an environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install numpy pandas matplotlib jupyter

# 3) Put Kaggle CSVs in a local folder, e.g. ./data/
#    (train.csv and test.csv)

# 4) Launch Jupyter and open the notebook
jupyter notebook
```

Update data paths in the notebook if running locally:

```python
# Kaggle paths
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
dataTest = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# Local paths
# data = pd.read_csv('./data/train.csv')
# dataTest = pd.read_csv('./data/test.csv')
```

---

## 🧠 Model

- **Architecture:** `Input (784)` → `Hidden (10, ReLU)` → `Output (10, Softmax)`
- **Loss:** Cross-entropy via softmax output (implicit in gradient derivation)
- **Optimization:** Vanilla gradient descent (`alpha = 0.10` by default)
- **Preprocessing:** Pixel values divided by 255

Key functions (as implemented in the notebook):

- `forward_prop(W1, b1, W2, b2, X)`  
- `backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)`  
- `update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)`  
- `gradient_descent(X, Y, alpha, iterations)`

---

## ▶️ Training

The notebook:
- Shuffles the data
- Uses the first **1000** examples as `dev`
- Trains on the remainder
- Prints progress every 100 iterations

Example training call:

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.10, iterations=100)
```

---

## ✅ Evaluation

Use the **dev split** for accuracy:

```python
_, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
preds_dev = np.argmax(A2_dev, axis=0)
accuracy = np.mean(preds_dev == Y_dev)
print(f"Dev accuracy: {accuracy:.4f}")
```

> The notebook includes a helper `test_prediction(...)` that samples examples for quick checks. Ensure you test against **labeled data** (e.g., `X_dev`, `Y_dev`). The Kaggle `test.csv` does **not** include labels.

---

## 📈 Results (baseline)

This is a deliberately tiny network (10 hidden units) with simple GD and no regularization/augmentation — it’s meant for learning, not leaderboard chasing. Expect a modest dev accuracy. To improve:

- Increase hidden layer size (e.g., 64–256)
- Use better initialization (He/Xavier)
- Add regularization (L2 / dropout)
- Switch to mini-batches + momentum/Adam
- Train longer and tune the learning rate

I can add an **Experiments** section with your actual dev accuracy once you run it.

---

## 🧪 Submitting to Kaggle

To create a submission:

1. Run the trained model on `test.csv`.
2. Generate predictions (0–9) for each row.
3. Save a CSV with **two columns**: `ImageId,Label` where `ImageId` starts at 1.

I can add a small cell in the notebook to export `submission.csv` on request.

---

## ⚠️ Notes & Known Limitations

- **Don’t evaluate on `test.csv`:** Kaggle test has no labels. Use `dev` from the training set for accuracy.
- **Numerical stability:** `softmax` can be stabilized by subtracting `Z.max(axis=0)` before `exp`.
- **Bias terms:** Summations for `db1/db2` in backprop should keep dimensions as column vectors; the current implementation is simplified.

If you want, I can harden the implementation (vectorized, numerically stable softmax, explicit cross-entropy) and add unit tests.

---

## 📚 Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `jupyter`

Quick install:

```bash
pip install -r requirements.txt
```

(Ask me to generate `requirements.txt` if you’d like versions pinned.)

---

## 🤝 Contributing

PRs welcome! Ideas:
- Refactor into a `src/` package with tests
- Add training curves and confusion matrix
- Add CLI (`train.py`, `eval.py`, `predict.py`)
- Export `submission.csv` helper

---

## 📄 License

MIT (or your preferred license). Say the word and I’ll add the file.

---

## 🙏 Acknowledgments

- Kaggle **Digit Recognizer** competition  
- Yann LeCun et al. for MNIST
# Digit-Recognizer
