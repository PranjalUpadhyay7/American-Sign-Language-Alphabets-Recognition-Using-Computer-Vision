<div align="center">
  <h1 style="color: #2E3440;">🖐️ ASL Alphabet Recognition Using Computer Vision</h1>
  <p><strong><em>An end-to-end Deep Learning pipeline implementing a custom Residual Convolutional Neural Network (CNN) to classify 29 American Sign Language (ASL) gestures with high precision.</em></strong></p>
</div>

---

## Executive Summary
Built for scalability and robust real-world inference, this project processes RGB hand sign images to predict ASL alphabets (A-Z) plus special gestures (Space, Delete, Nothing). By leveraging a **Hybrid CNN with Custom Residual Blocks**, the architecture effectively solves the vanishing gradient problem in deep networks, allowing for deep feature extraction while maintaining a lightweight footprint through **Global Average Pooling (GAP)**.

**Tech Stack:** Python, TensorFlow / Keras, OpenCV, NumPy, Matplotlib, Seaborn

---

## 🧠 System Architecture

The overarching pipeline spans from data ingestion and augmentation to inference. 

```mermaid
flowchart TD
    A[Raw ASL Directory] -->|ImageDataGenerator| B(Augmentation Engine)
    B -->|Rotation, Shear, Zoom, Shift| C{Hybrid ResNet CNN}
    C -->|Conv + ResBlocks| D[Deep Feature Extraction]
    D -->|GAP + Dense Layers| E[29-Class Softmax]
    E --> F[Predicted ASL Letter]
    
    classDef input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef model fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    
    class A input;
    class B,D process;
    class C,E model;
    class F output;
```

---

## 🏗️ Model Blueprint: Hybrid ResNet CNN

The model deviates from traditional sequential CNNs by implementing **identity shortcuts (Residual Blocks)** accompanied by Batch Normalization. This topology promotes smoother gradient flow and robust convergence.

<details open>
<summary><b>Click to expand Model Flowchart</b></summary>

```mermaid
flowchart TD
    IN((Input: 64x64x3 RGB)) --> C1[VGG-style Conv Block 1<br/>32 Filters + MaxPool + Dropout]
    C1 --> R1[Residual Block 1<br/>Shortcut + Batch Norm]
    
    R1 --> C2[VGG-style Conv Block 2<br/>64 Filters + MaxPool + Dropout]
    C2 --> R2[Residual Block 2<br/>Shortcut + Batch Norm]
    
    R2 --> C3[VGG-style Conv Block 3<br/>128 Filters + MaxPool + Dropout]
    C3 --> R3[Residual Block 3<br/>Shortcut + Batch Norm]
    
    R3 --> GAP[[Global Average Pooling 2D]]
    GAP --> D1[Dense 256 + Dropout: 0.5]
    D1 --> D2[Dense 128 + Dropout: 0.3]
    D2 --> OUT((Output Dense: 29 Softmax))
    
    classDef layer fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000
    classDef resblock fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef dense fill:#ffe0b2,stroke:#f57c00,stroke-width:2px,color:#000
    
    class IN,C1,C2,C3,GAP layer;
    class R1,R2,R3 resblock;
    class D1,D2,OUT dense;
```
</details>

### Model Highlights:
1. **Data Augmentation:** Real-time generation using `ImageDataGenerator` (15° rotation, 10% shift, zoom, shear) guarantees the model learns positional invariances.
2. **Residual Blocks:** Custom implementation mapping input tensors (`shortcut`) to the output of 2 stacked Convolutions. 
3. **GAP Layer:** Flattens 2D features to 1D while aggressively combating overfitting by taking spatial averages.

---

## 📦 Repository Structure

```text
📁 ASL-Alphabets-Recognition
│
├── 📄 train.ipynb           # Model definition, Data Pipeline, Training, Evaluation
├── 📄 best_model.keras      # Serialized weights of the best performing epoch
├── 📄 Pranjal_Upadhyay_22244.pdf # Detailed project report
└── 📄 Presentation_PranjalUpadhyay_22244.pdf # Presentation slides
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required ML stack:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 2. Dataset Setup
Download the standard **ASL Alphabet Dataset** and organize it into the following structure:
* `ASL_dataset/asl_alphabet_train/` - (29 subfolders)
* `ASL_dataset/asl_alphabet_test/` - (29 subfolders)
* `ASL_dataset/test/` - (Validation images for real-world testing)

### 3. Usage & Inference
You can train the model from scratch or use the provided weights (`best_model.keras`) to run immediate inferences. Open the `train.ipynb` notebook to:
* **Train a new model:** The notebook is configured with `Adam(0.001)` and `EarlyStopping`.
* **Evaluate Performance:** Visualizes the `Confusion Matrix` and outputs a dense `Classification Report`. 
* **Run live predictions:** Iterates through `ASL_dataset/test` to render images with their predicted ASL overlay.
