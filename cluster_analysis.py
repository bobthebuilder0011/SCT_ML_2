"""
Advanced K-Means Clustering Analysis with PCA
This script performs dimensionality reduction and clustering for customer segmentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import argparse
import os

def run_analysis(input_file='Mall_Customers.csv', n_clusters=5):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load data
    df = pd.read_csv(input_file)
    print("=" * 60)
    print("ADVANCED CUSTOMER SEGMENTATION - K-MEANS & PCA")
    print("=" * 60)
    
    # Preprocessing
    # Identify numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    features = [col for col in numeric_df.columns if 'id' not in col.lower()]
    X = numeric_df[features].fillna(numeric_df[features].mean())
    
    print(f"\nAnalyzing features: {', '.join(features)}")
    print(f"Dataset Shape: {df.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    # Elbow and Silhouette Method
    print("\nOptimizing K...")
    inertias = []
    scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        scores.append(silhouette_score(X_scaled, labels))
        
    optimal_k = K_range[np.argmax(scores)]
    print(f"Suggested Optimal K (Silhouette): {optimal_k}")

    # Final Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualization
    plt.figure(figsize=(16, 12))
    
    # 1. PCA Scatter Plot
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
    plt.title('Clusters in PCA-Reduced Space')
    plt.grid(True, alpha=0.3)
    
    # 2. Elbow Plot
    plt.subplot(2, 2, 2)
    plt.plot(K_range, inertias, 'bo-', linewidth=2)
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True, alpha=0.3)
    
    # 3. Features Heatmap by Cluster
    plt.subplot(2, 2, 3)
    cluster_avg = df.groupby('Cluster')[features].mean()
    sns.heatmap(cluster_avg, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Average Feature Values by Cluster')
    
    # 4. Cluster Distribution
    plt.subplot(2, 2, 4)
    df['Cluster'].value_counts().sort_index().plot(kind='bar', color=sns.color_palette('viridis', n_clusters))
    plt.title('Customer Count per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('cluster_report.png', dpi=300)
    print("\nReport saved as 'cluster_report.png'")
    
    # Detailed Report
    print(f"\n{'='*60}")
    print(f"CLUSTER PROFILES (K={n_clusters})")
    print(f"{'='*60}")
    
    for i in range(n_clusters):
        c_data = df[df['Cluster'] == i]
        print(f"\n[Cluster {i}] - {len(c_data)} customers ({len(c_data)/len(df)*100:.1f}%)")
        for f in features:
            print(f"  Avg {f:20}: {c_data[f].mean():.1f}")
            
    # Save output
    df.to_csv('segmented_customers.csv', index=False)
    print(f"\nData saved as 'segmented_customers.csv'")
    print(f"Final Silhouette Score: {silhouette_score(X_scaled, df['Cluster']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customer Segmentation')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters')
    parser.add_argument('--input', type=str, default='Mall_Customers.csv', help='Input CSV file')
    args = parser.parse_args()
    
    run_analysis(args.input, args.k)
