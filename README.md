<div align="center">

# 🛣️ Pothole Detection System
### *AI-Powered Road Safety & Infrastructure Monitoring*

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)
![Keras](https://img.shields.io/badge/Keras-ResNet50-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**Pothole Detection System** is a deep learning solution designed to identify road hazards from images automatically. 
It leverages transfer learning with **ResNet50** to classify road surfaces as either **Normal** or containing a **Pothole** with high accuracy.

> **Is the road ahead safe? Detect potholes instantly to prevent accidents and vehicle damage.**

</div>

---

# 🚀 What Problem Does It Solve?

Poor road conditions are a major cause of accidents and vehicle wear-and-tear globally.

| Problem | Why it matters |
| :--- | :--- |
| **Safety Hazards** | Potholes cause accidents, especially for two-wheelers. |
| **Vehicle Damage** | Regular impact leads to costly suspension and tyre repairs. |
| **Manual Inspection** | Traditional road monitoring is slow, labor-intensive, and expensive. |
| **Delayed Repairs** | Authorities often lack real-time data on road conditions. |

**Pothole Detection System** addresses this by:
1.  **Automated Classification**: Instantly detecting potholes from camera feeds or images.
2.  **High Accuracy**: Utilizing a pre-trained ResNet50 model fine-tuned for road texture analysis.
3.  **Accessible Interface**: Providing a simple web app for users and authorities to test images.

---

# 🧩 System Architecture

**Architecture Explained:**
*   **Input**: Road surface images (camera feed or upload).
*   **Preprocessing**: Resizing to 224x224, normalization, and augmentation.
*   **Deep Learning Model**: 
    *   **Base**: ResNet50 (Pre-trained on ImageNet)
    *   **Head**: Global Average Pooling + Dense Layers + Dropout for regularization.
    *   **Output**: Binary Classification (Normal vs. Pothole).
*   **Interface**: Streamlit Web App for real-time inference.

---

# 📁 Project Structure

```
Pothole-Detection/
│
├── dataset/                    # Training and Validation Data
│   ├── train/
│   ├── val/
│   └── test/
│
├── model/
│   └── pothole_detection_model.h5  # Trained Keras Model
│
├── prediction_outputs/         # Saved predictions and evaluation graphs
│   ├── checked_*.jpg           # Annotated prediction images
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── testing/                    # Test images folder
│   ├── Plain_Test_*.jpg
│   └── Pothole_Test_*.jpg
│
├── 01_training.ipynb           # Model Training Notebook
├── 02_prediction_and_evaluation.ipynb # Batch Prediction & Metrics Notebook
├── app.py                      # Streamlit Web Application
├── perform_predictions.py      # Automated prediction script
├── README.md                   # Project Documentation
└── requirements.txt            # Python Dependencies
```

---

# 🧠 Machine Learning Performance

The model is trained on a curated dataset of road images.

### 🏆 Key Metrics
*   **Model Architecture**: ResNet50 (Transfer Learning)
*   **Test Accuracy**: **~98%**
*   **Precision/Recall**: High precision in detecting potholes, minimizing false positives.

> *The model effectively distinguishes between normal road textures, shadows, and actual potholes.*

---

# 💻 Setup & Usage

### 1️⃣ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/saishhhhhh/Pothole-Detection-CNN.git
cd Pothole-Detection
pip install tensorflow streamlit opencv-python matplotlib seaborn scikit-learn
```

### 2️⃣ Run the Web App

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Upload an image to see the prediction and confidence score!

### 3️⃣ Batch Prediction

To evaluate a folder of images and generate reports:

```bash
# Run the evaluation notebook or script
python perform_predictions.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Developed with ❤️ by Saish**

⭐ Star this repo if you find it useful for road safety innovation!

</div>
