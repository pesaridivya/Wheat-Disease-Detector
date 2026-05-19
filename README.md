# Wheat Disease Detector using Deep Learning

An AI-powered web application for detecting wheat leaf diseases using **CNN + Logistic Regression hybrid model**.  
The system classifies wheat leaf images into:

- Black Rust
- Brown Rust
- Yellow Rust
- Healthy Wheat

The project uses **PyTorch**, **MobileNetV3**, and **Flask** to provide fast and accurate disease prediction with treatment recommendations.

---

# Features

- Deep Learning based wheat disease detection
- Hybrid CNN + Logistic Regression model
- Upload wheat leaf image for prediction
- Disease confidence score visualization
- Treatment and prevention recommendations
- Responsive web interface
- Real-time prediction using Flask

---

#  Technologies Used

- Python
- Flask
- PyTorch
- Torchvision
- MobileNetV3
- Logistic Regression
- HTML
- CSS
- JavaScript

---

#  Model Performance

| Model | Accuracy |
|---|---|
| CNN | 93.97% |
| CNN + Random Forest | 95.76% |
| CNN + SVM | 96.38% |
| CNN + Logistic Regression | **96.51%** |

---
#  Dataset

The complete wheat leaf disease dataset contains **4000+ images** and is hosted on **Kaggle** .

## Download Dataset from Kaggle

https://www.kaggle.com/datasets/pesaridivya/wheat-disease

## Dataset Classes

- Black Rust
- Brown Rust
- Healthy Wheat
- Yellow Rust
# Installation

Follow these steps to run the Wheat Disease Detector project locally.

---

##  Clone the Repository

```bash
git clone https://github.com/your-username/Wheat-Disease-Detector.git
cd Wheat-Disease-Detector
```
Install Required Libraries
```bash
pip install -r requirements.txt
```
Run the Applicaion
```bash
python app.py
```
