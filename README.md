# Federated Learning on Iris Dataset with PyTorch

This project implements a **Federated Learning** simulation using the Iris dataset and a simple neural network model built in **PyTorch**. The goal is to simulate distributed training across multiple clients, aggregate their models, and evaluate global model accuracy across multiple rounds.

## 🧠 Project Highlights

- Uses the classic **Iris dataset** from `sklearn.datasets`.
- Implements **federated learning** with 3 virtual clients.
- Trains a **simple neural network** with two fully connected layers.
- Aggregates client models using both:
  - **Simple averaging**
  - A customized **softmax-weighted approach** based on model deviation.
- Evaluates the performance of the **global model** across rounds and iterations.
- Visualizes **model accuracy** trends over training rounds.

## 📂 Project Structure

- `load_iris` & `shuffle` — Dataset preparation
- `SimpleNet` — Basic neural network model
- Client-side training loop with `DataLoader`
- Federated aggregation logic
- Evaluation of local and global model accuracies
- Repeated training over multiple iterations for robustness

## 🚀 How It Works

1. **Dataset Preparation**:
   - Loads and shuffles the Iris dataset.
   - Splits data into training and test sets.
   - Splits training data across 3 clients.

2. **Model Definition**:
   - A two-layer fully connected neural network is defined using PyTorch.

3. **Federated Training**:
   - Each client receives the global model weights.
   - Clients train locally on their data using SGD and cross-entropy loss.
   - Local models are aggregated on the server using either:
     - Equal averaging (FedAvg)
     - Custom softmax-weighted aggregation based on parameter deviations.

4. **Evaluation**:
   - The global model is tested after each round using test data.
   - Accuracy is recorded and averaged over multiple runs for analysis.

5. **Visualization**:
   - Accuracy of the global model is plotted across rounds.

## 🛠️ Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib
- numpy

Install requirements via:

```bash
pip install torch scikit-learn matplotlib numpy
