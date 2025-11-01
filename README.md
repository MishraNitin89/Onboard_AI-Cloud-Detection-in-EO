# ClouDSen Project – Onboard Cloud Detection for Sentinel-2 Imagery

## 🌍 Overview
This project develops a lightweight deep learning model for **cloud detection in Sentinel-2 satellite images**, optimized for **edge deployment** (Jetson Nano / Raspberry Pi).  
The model uses a **MobileNetV3-U-Net** architecture trained on the **CloudSEN12+ dataset**.

---

## 🧩 Project Modules
1. **Data Handling** – Read and preprocess Sentinel-2 tiles and cloud masks.
2. **EDA** – Visualize spectral bands, mask distribution, and cloud coverage.
3. **Model Architecture** – Implement MobileNetV3-U-Net in TensorFlow.
4. **Training** – Train and evaluate segmentation performance (IoU, F1).
5. **Optimization** – Prune and quantize using TensorFlow Model Optimization Toolkit.
6. **Edge Deployment** – Benchmark TFLite INT8 model on Jetson Nano / Raspberry Pi.

---

## ⚙️ Setup
```bash
pip install -r requirements.txt
```

---

## 📁 Structure
```
clouden-project/
├── notebooks/           # Jupyter notebooks for EDA, training, quantization
├── scripts/             # Core python scripts (data, model, training)
├── results/             # Metrics, plots, logs
├── models/              # Saved models (.h5, .tflite)
├── data/                # Sample input tiles (small)
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🧠 References
- Ye et al. (2024). *CloudSEN12+: A large-scale expert-labeled Sentinel-2 cloud and cloud-shadow dataset.*
- Φ-Sat-1 ESA Mission – Onboard AI for Earth Observation.
