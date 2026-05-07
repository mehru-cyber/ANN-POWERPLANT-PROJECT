 ANN-POWERPLANT-PROJECT
PyTorch ANN that predicts power plant energy output (MW) from ambient temperature, vacuum, pressure &amp; humidity — full ML pipeline with StandardScaler, mini-batch training, model checkpointing &amp; R² evaluation.
 Power Plant Energy Output Prediction — ANN Regression

 Predicting electrical energy output of a combined-cycle power plant using a custom Artificial Neural Network built with PyTorch.
 
1.Project Overview

Combined-cycle power plants (CCPPs) use both gas and steam turbines to generate electricity. Their output depends on several ambient environmental conditions. This project builds and trains a **multi-layer Artificial Neural Network (ANN)** from scratch in PyTorch to predict the **net hourly electrical energy output (PE)** of a CCPP, given four ambient sensor readings.

This is a real-world **regression problem** that demonstrates the full ML pipeline: data exploration → preprocessing → model design → training with validation → evaluation.



2.Dataset



 Source : UCI Machine Learning Repository — [CCPP Dataset](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) |
 Samples : 9,568 hourly records 
 Features : 4 continuous input variables 
 Target : Net hourly energy output (PE) in MW 

3. Features

 Column | Full Name | Unit | Description 

 `AT` | Ambient Temperature | °C | Hourly avg. temperature 
 `V` | Exhaust Vacuum | cm Hg | Vacuum in steam turbine 
 `AP` | Ambient Pressure | millibar | Hourly avg. pressure 
 `RH` | Relative Humidity | % | Hourly avg. humidity 
 `PE` | Net Energy Output *(target)* | MW | What we predict 

4. Statistical Summary

 AT | V | AP | RH | PE 

 Mean | 19.65 | 54.31 | 1013.26 | 73.31 | 454.37 |
 Std | 7.45 | 12.71 | 5.94 | 14.60 | 17.0|
 Min | 1.81 | 25.36 | 992.89 | 25.56 | 420.26 
 Max | 37.11 | 81.56 | 1033.30 | 100.16 | 495.76 


5. Model Architecture

A custom feedforward ANN built with **PyTorch's `nn.Module`**:

Input Layer  →  4 features
     ↓
Hidden Layer 1  →  6 neurons  +  ReLU
     ↓
Hidden Layer 2  →  6 neurons  +  ReLU
     ↓
Output Layer  →  1 neuron (predicted PE)
```

- Loss Function: Mean Squared Error (MSELoss)
- Optimizer Adam (adaptive learning rate)
- Epochs: 100
- Batch Size: 32
- Best model saved via checkpoint using `torch.save()

---

6. ML Pipeline 

```
Raw CSV Data
    
    ▼
Exploratory Data Analysis (null checks, shape, stats)
    
    ▼
Train/Test Split  →  80% train / 20% test  (stratified by random_state=42)
    
    ▼
Feature Scaling  →  StandardScaler (fit on train, transform both)
    
    ▼
PyTorch Tensors + DataLoader (batch_size=32, shuffle=True)
    
    ▼
ANN Training with per-epoch train & validation loss tracking
    
    ▼
Best Model Checkpoint (saved when val loss improves)
    
    ▼
Evaluation  →  MSE + R² Score  +  Predicted vs Actual DataFrame
```

---

7.Results

 Metric | Value 

 Loss Function | MSE (Mean Squared Error) 
 Evaluation Metric | R² Score 
 Training tracked across | 100 epochs 
Validation strategy | Hold-out test set (20%) 

> Loss curves (Training vs Validation) are plotted using `matplotlib` to diagnose overfitting and monitor convergence.



8. Tech Stack

 Tool | Purpose

 `Python 3` | Core language 
 `pandas` | Data loading & manipulation 
 `numpy` | Numerical operations 
 `scikit-learn` | Train/test split, StandardScaler, R² score 
 `PyTorch` | ANN model, training loop, checkpointing 
 `matplotlib` | Loss curve visualization 

---

9. Repository Structure

```
 powerplant-ann-regression/
 ┣  ANN_Regression.ipynb     # Main notebook — full pipeline
 ┣  powerplant_data.csv      # Dataset (9,568 records)
 ┣  best_model.pt            # Saved best model weights
 ┗  README.md                # You're here!
```

---

10. Getting Started

Prerequisites
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Run the Notebook
```bash
git clone https://github.com/YOUR_USERNAME/powerplant-ann-regression.git
cd powerplant-ann-regression
jupyter notebook ANN_Regression.ipynb
```

---

11.Key Learnings

- Designed and trained a **custom neural network** from scratch in PyTorch without using high-level wrappers like Keras
- Implemented a **manual training loop** with gradient zeroing, forward pass, backpropagation, and parameter updates
- Applied **model checkpointing** to preserve the best-performing model during training
- Used **StandardScaler** correctly — fit only on training data to prevent data leakage
- Evaluated performance using both **MSE** and **R² score** for comprehensive regression analysis



 Contact

Made by MEHRU NISA — feel free TO reach out via GitHub Issues.



⭐ Star this repo if you found it helpful!
