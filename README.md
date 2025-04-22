
# 🧠 CNN Waste Segregation – Deep Learning Project

A CNN-based image classification project focused on automating the segregation of waste into predefined categories. Implemented in Jupyter Notebook using TensorFlow/Keras, this project evaluates three different CNN models and identifies the most effective architecture based on accuracy and loss metrics.

---

## 📂 Project Overview

The goal of this project is to develop a robust waste classification model to support automatic waste segregation systems. By training a CNN on labeled waste images, the model predicts the correct category of the waste item (e.g., biodegradable, recyclable, etc.).

---

## 📊 Dataset Summary

- The dataset is divided into `train`, `test`, and `validation` folders.
- Each folder contains images sorted into category-based subdirectories.
- Images were resized to (64x64) and normalized for CNN processing.

---

## 🏗️ Model Architectures Analyzed

### 🔹 **Model 1: Basic CNN**
- 2 Convolutional layers + MaxPooling
- 1 Dense layer with 64 units
- Output layer with Softmax activation
- **Performance:**
  - Training Accuracy: 86%
  - Validation Accuracy: 84%
  - Observations: Basic model with underfitting signs; good as baseline.

### 🔹 **Model 2: Deeper CNN with Dropout**
- 3 Convolutional layers + MaxPooling
- Dropout layers to prevent overfitting
- Dense layer with 128 units
- **Performance:**
  - Training Accuracy: 91%
  - Validation Accuracy: 89%
  - Observations: Improved accuracy with better generalization; reduced overfitting due to Dropout.

### 🔹 **Model 3: Advanced CNN with Batch Normalization**
- 3 Convolutional layers with Batch Normalization
- MaxPooling and Dropout for regularization
- Dense layers with 256 and 128 units
- **Performance:**
  - Training Accuracy: 94%
  - Validation Accuracy: 92%
  - Observations: Best performance among all models; faster convergence and good generalization.

---

## 📌 Conclusion

- **Model 3** demonstrated the best performance in terms of both training and validation accuracy, with strong generalization and minimal overfitting.
- Batch Normalization and Dropout significantly improved learning stability and reduced variance.
- The project showcases the importance of tuning architecture depth and regularization in CNN design.

✅ **Final Model Chosen:** Model 3  
📈 **Best Accuracy Achieved:** 94% (training), 92% (validation)

---

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Ensure the dataset is structured correctly with train/val/test directories.
3. Install dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook CNN_Waste_Segregation_SravanaSanka.ipynb
   ```
5. Execute cells sequentially to train and evaluate the models.

---

## 📚 Dependencies

- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🔮 Future Scope

- Deploy final model using Streamlit or Flask
- Integrate with a camera feed for real-time predictions
- Extend classification to include more granular waste categories
