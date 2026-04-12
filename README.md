# ClusterCraft AI: Advanced Customer Segmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)
![PCA](https://img.shields.io/badge/Feature-PCA-purple.svg)

An advanced machine learning platform for customer segmentation using **K-Means Clustering** and **Principal Component Analysis (PCA)**. This project provides a robust solution for businesses to discover hidden patterns in customer data through interactive dimensionality reduction and high-dimensional visualization.

---

## 🚀 Key Features

### 🧠 Advanced Analytics
- **Multi-Feature Clustering**: Automatically handles any number of numeric features.
- **Dimensionality Reduction**: Implements **PCA** (Principal Component Analysis) to project high-dimensional data into 2D and 3D space.
- **Dynamic Optimization**: Real-time Silhouette Score and Inertia (WCSS) calculation.
- **Automated Preprocessing**: Intelligent handling of missing values and feature scaling.

### 🌐 Modern Web Dashboard
- **Universal File Support**: Upload any CSV dataset for instant clustering analysis.
- **Interactive 3D Visualization**: Explore customer segments in a rotatable 3D PCA space.
- **Responsive Design**: Dark-themed, glassmorphism UI built with Tailwind CSS.
- **Real-time Metrics**: Live updates of clustering quality as you adjust parameters.

### 💻 CLI Power Tool
- **Automated Reporting**: Generates high-resolution statistical reports (`cluster_report.png`).
- **Batch Processing**: Scriptable interface for large-scale data segmentation.

---

## 🛠️ Technology Stack

- **Backend:** Flask, Scikit-Learn, Pandas, NumPy
- **Frontend:** Tailwind CSS, Plotly.js, FontAwesome
- **ML Algorithms:** K-Means Clustering, Principal Component Analysis (PCA)

---

## 🚦 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launch the Platform
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000` to access the interactive dashboard.

### 3. Run Standalone Analysis
```bash
python cluster_analysis.py --k 5 --input Mall_Customers.csv
```

---

## 📊 How It Works

1. **Data Ingestion**: Upload a CSV file. The system automatically identifies numeric features and handles preprocessing.
2. **Feature Scaling**: Uses `StandardScaler` to ensure all features contribute equally to the distance metrics.
3. **PCA Projection**: Reduces the feature space to 2D and 3D components for visualization while retaining maximum variance.
4. **K-Means Clustering**: Partitions customers into K distinct groups based on feature similarity.
5. **Evaluation**: Calculates the **Silhouette Score** (range -1 to 1) to validate the quality of the segments.

---

## 📂 Project Structure

- `app.py`: Enhanced Flask server with PCA and file upload endpoints.
- `cluster_analysis.py`: Professional CLI tool for generating visual reports.
- `templates/index.html`: Modern, single-page application dashboard.
- `uploads/`: Temporary storage for user-uploaded datasets.

---

## 📈 Future Roadmap

- [ ] Implementation of DBSCAN and Hierarchical Clustering.
- [ ] Automated "Optimal K" recommendation engine.
- [ ] Exportable PDF reports and segment summaries.
- [ ] User authentication and saved analysis history.

---
**Author:** bobthebuilder0011  
**License:** MIT
