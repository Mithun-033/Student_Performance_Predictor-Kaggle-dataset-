# Student Performance Predictor

A simple neural network model that predicts student scores using a custom 2-layer ANN built with minimal PyTorch abstractions. Trained on a Kaggle dataset containing multiple student performance features.

---

## 📁 Project Files

- **data.csv** — Raw dataset used for training and testing  
- **model_weights.pth** — Pre-trained weights and biases  
- **scaler_values.pth** — Mean & standard deviation values for normalization  
- **predict.py** — Script to load the model and generate custom predictions  

Both pre-trained files achieve 99%+ accuracy with the provided evaluation setup.

---

## 🚀 How to Use

```bash
python predict.py
