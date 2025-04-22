
# 🧠 CNN Waste Segregation

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

## **With Custom Model 1** 🛠️

- **Custom Model 1** achieved an overall accuracy of **61%**, showing solid improvement over the baseline. 📈
- **Cardboard** and **Plastic** had the best performance, with **F1-scores** of **0.73** and **0.69**, respectively. 📦🛢️
- Lower scores for **Other**, **Paper**, and **Glass** suggest potential confusion or class overlap, which may require further attention. 🚧📉

---

## **With Custom Model 2** 🛠️💡

- **Custom Model 2** achieved a higher overall accuracy of **66%**, improving across most classes compared to Model 1. 📊🔝
- Top-performing classes include **Cardboard** (**F1-score 0.78**) and **Plastic** (**F1-score 0.72**), showing strong consistency. 📦🛢️
- Performance for **Other** and **Paper** remains moderate, indicating room for refinement in distinguishing these categories. ✋📜

---

## **With MobileNet-V2 Model** 📱🚀

- The **MobileNet V2** customized model achieved an impressive **84% accuracy**, significantly outperforming previous models. 🌟📈
- High **F1-scores** across all classes, especially **Cardboard** (**0.92**), **Metal** (**0.90**), and **Food Waste** (**0.86**), reflect strong generalization. 📦🔩🍲
- Even lower-performing classes like **Paper** and **Other** show solid improvements, making this model highly reliable overall. 📑🔄


---

## 📌 Conclusion

## **Findings from the Data**

- The dataset consists of images categorized into seven classes: **Metal**, **Other**, **Glass**, **Food Waste**, **Paper**, **Plastic**, and **Cardboard**. 🗑️🖼️

- As shown in the class distribution chart above, the dataset is **imbalanced**, with **Plastic** having the highest number of samples, while **Cardboard** has the fewest. ⚖️📊

- This imbalance could lead to biased model predictions, particularly favoring the majority class (**Plastic**). ⚠️🔍

- To address this, **data augmentation** has been applied to balance the dataset. 🔄✨

- Some classes (such as **Food Waste** and **Plastic**) exhibit considerable variation in appearance, which may complicate classification. 🍲🛢️

- **Visually similar materials**, like **Plastic** and **Glass**, may confuse the model due to overlapping textures or colors. 🥤🍷

- Variations in **image quality** or **lighting conditions** may introduce noise, potentially affecting model performance. 🌥️📸

---

## **Model Training and Results**

- A **Convolutional Neural Network (CNN)** was trained from scratch, followed by using a predefined architecture, and ultimately with **augmentation techniques**. 🤖📈

- The input image size was resized to **(128, 128, 3)** to preserve key features. 🖼️🔍

  **Note:** I attempted to use the dimension **(224, 224, 3)**, but Google Colab crashed multiple times. 😵💻

- The model with predefined layers exhibited **significant improvement** in accuracy and **reduced overfitting**. 📊🔧

- **Callbacks** such as **EarlyStopping**, **ModelCheckpoint**, and **ReduceLROnPlateau** were employed to stabilize and optimize the training process. ⏱️💡

---

## **Key Insights**

- **Image resolution** and **class balance** have a major impact on the performance of the classification model. 🖼️⚖️

- Effective use of **callbacks** not only helped preserve the best-performing model but also prevented unnecessary training cycles, optimizing the overall training time. ⏳🎯


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
- Collab

##📬 Contact
For any queries or collaboration: [Sravana Kumar Sanka]
📧 Email: sravan.sanka97@gmail.com
🔗 GitHub: [github.com/Sravansanka/CNN_Waste_Segregation](https://github.com/Sravansanka/CNN_Waste_Segregation)]
