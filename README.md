# 🏙️ Urban Sound Classification Ensemble

This project implements an end-to-end Machine Learning pipeline to classify environmental urban sounds (e.g., sirens, dog barks, drilling). It utilizes a **Hybrid Ensemble Architecture** combining classical Machine Learning (SVM) with Deep Learning (ANN & ResNet) to achieve high-accuracy predictions via majority voting.

## 🗂️ Dataset
The project uses the **UrbanSound8K** dataset, consisting of 8,732 labeled sound excerpts (<= 4s) across 10 classes:
* Air Conditioner, Car Horn, Children Playing, Dog Bark, Drilling, Engine Idling, Gun Shot, Jackhammer, Siren, Street Music.

## 🏗️ System Architecture

The system processes audio inputs through two distinct feature extraction pipelines to leverage both statistical and spatial characteristics of sound.

### 1. Feature Extraction Pipelines
* **Pipeline A: Flat Features (for SVM & ANN)**
    * **Technique:** Mel-Frequency Cepstral Coefficients (MFCCs).
    * **Preprocessing:** Standardization (`StandardScaler`) to normalize distribution.
    * **Dimensionality Reduction:** Principal Component Analysis (**PCA**) applied for the SVM to retain 95% variance.
* **Pipeline B: Spatial Features (for ResNet)**
    * **Technique:** Mel-Spectrograms converted to Log-Decibel scale.
    * **Preprocessing:** Resizing/Padding to a fixed 128x128 resolution to treat audio as an image.

### 2. The Models
Three distinct architectures were trained and optimized:

| Model | Type | Architecture Details | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| **Custom ResNet** | CNN (Deep Learning) | Custom-built Residual Blocks with Skip Connections to prevent signal degradation. Trained from scratch (not pre-trained). | **~92.8%** |
| **Optimized SVM** | Classical ML | Support Vector Machine with RBF Kernel. Hyperparameters tuned via GridSearchCV. | **~92.1%** |
| **ANN** | MLP (Deep Learning) | Dense Neural Network with Dropout layers to prevent overfitting on flat vectors. | **~89.5%** |

### 3. Ensemble Deployment (Streamlit)
The final deployment (`app.py`) integrates all three models using a **Majority Voting Mechanism**:
1.  **Consensus:** If 2 or more models predict the same class, that class is selected.
2.  **Tie-Breaker:** If all 3 models disagree, the system defaults to the **ResNet** prediction (as it demonstrated the highest individual accuracy during training).

