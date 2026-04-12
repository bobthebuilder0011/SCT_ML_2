import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to store current dataset and features
CURRENT_DATA = pd.read_csv('Mall_Customers.csv')
FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    """Clean and prepare data for clustering"""
    processed_df = df.copy()
    
    # Identify numeric columns for clustering
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude ID columns if present
    clustering_features = [col for col in numeric_cols if 'id' not in col.lower()]
    
    if not clustering_features:
        return None, None, None
        
    X = processed_df[clustering_features]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, clustering_features, processed_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global CURRENT_DATA, FEATURES
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            new_df = pd.read_csv(filepath)
            X_scaled, features, processed_df = preprocess_data(new_df)
            
            if X_scaled is None:
                return jsonify({'error': 'No numeric features found for clustering'}), 400
                
            CURRENT_DATA = new_df
            FEATURES = features
            return jsonify({
                'message': 'File uploaded successfully',
                'features': FEATURES,
                'rows': len(new_df)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/data')
def get_data():
    """Return current customer data"""
    return jsonify(CURRENT_DATA.to_dict(orient='records'))

@app.route('/api/cluster', methods=['POST'])
def cluster():
    """Perform K-means clustering with PCA and Silhouette Score"""
    data = request.get_json() or {}
    n_clusters = int(data.get('n_clusters', 5))
    
    X_scaled, features, processed_df = preprocess_data(CURRENT_DATA)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate Silhouette Score
    score = silhouette_score(X_scaled, cluster_labels) if n_clusters > 1 else 0
    
    # PCA for 2D and 3D visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    pca_3d = PCA(n_components=3) if len(features) >= 3 else None
    X_pca_3d = pca_3d.fit_transform(X_scaled) if pca_3d else None

    # Prepare results
    result_df = CURRENT_DATA.copy()
    result_df['Cluster'] = cluster_labels
    result_df['pca_x'] = X_pca_2d[:, 0]
    result_df['pca_y'] = X_pca_2d[:, 1]
    
    if X_pca_3d is not None:
        result_df['pca_3d_x'] = X_pca_3d[:, 0]
        result_df['pca_3d_y'] = X_pca_3d[:, 1]
        result_df['pca_3d_z'] = X_pca_3d[:, 2]

    # Prepare cluster summary
    clusters = []
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]

    for i in range(n_clusters):
        cluster_data = result_df[result_df['Cluster'] == i]
        cluster_info = {
            'id': i,
            'name': f'Cluster {i + 1}',
            'color': colors[i % len(colors)],
            'count': len(cluster_data),
            'customers': cluster_data.to_dict(orient='records')
        }
        
        # Add averages for numeric features
        for feat in features:
            cluster_info[f'avg_{feat}'] = round(cluster_data[feat].mean(), 2)
            
        clusters.append(cluster_info)

    return jsonify({
        'clusters': clusters,
        'silhouette_score': round(float(score), 4),
        'inertia': round(float(kmeans.inertia_), 2),
        'features': features,
        'explained_variance_ratio': pca_2d.explained_variance_ratio_.tolist()
    })

@app.route('/api/elbow')
def elbow():
    """Calculate metrics for different k values"""
    X_scaled, _, _ = preprocess_data(CURRENT_DATA)
    k_range = range(2, 11)
    inertias = []
    scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(round(float(kmeans.inertia_), 2))
        scores.append(round(float(silhouette_score(X_scaled, kmeans.labels_)), 4))

    return jsonify({
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': scores
    })

@app.route('/api/stats')
def stats():
    """Return dataset statistics"""
    numeric_df = CURRENT_DATA.select_dtypes(include=[np.number])
    stats_data = {
        'total_customers': len(CURRENT_DATA),
        'columns': CURRENT_DATA.columns.tolist(),
        'numeric_stats': {}
    }
    
    for col in numeric_df.columns:
        stats_data['numeric_stats'][col] = {
            'mean': round(float(numeric_df[col].mean()), 2),
            'min': float(numeric_df[col].min()),
            'max': float(numeric_df[col].max())
        }
        
    return jsonify(stats_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
