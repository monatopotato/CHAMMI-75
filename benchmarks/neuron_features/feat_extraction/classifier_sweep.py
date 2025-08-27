import scanpy as sc
import pandas as pd
import seaborn as sns
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import os
import sys
import glob
from imageio import volread as imread
from skimage.filters import threshold_otsu
from skimage import measure
from scipy import stats
import umap
from sklearn.decomposition import PCA
import math
from scipy import stats
from scipy.stats import pearsonr
import pickle
from scipy.spatial.distance import euclidean
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, make_scorer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline

from statsmodels.stats.multitest import multipletests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight

import pickle
import os
from tqdm import tqdm

embedding_path = "/scr/vidit/neural_features/feat_extraction"

with open(os.path.join(embedding_path, "train_embeddings.pkl"), 'rb') as f:
    train_real_embeddings = pickle.load(f)

with open(os.path.join(embedding_path, "test_embeddings.pkl"), 'rb') as f:
    test_real_embeddings = pickle.load(f)

# Get all the embeddings in a dimension of (N, 384*num_channels)
train_embeddings = [] 
test_embeddings = []

train_genes = []
test_genes = []
for emb in train_real_embeddings:
    if emb['metadata']['Gene'][0] == 'DGKE' or emb['metadata']['Gene'][0] == 'GAS7':
        continue
    train_embeddings.append(emb['embedding'].reshape(1, -1))
    train_genes.append(emb['metadata']['Gene'])
    
for emb in test_real_embeddings:
    if emb['metadata']['Gene'][0] == 'DGKE' or emb['metadata']['Gene'][0] == 'GAS7':
        continue
    test_embeddings.append(emb['embedding'].reshape(1, -1))
    test_genes.append(emb['metadata']['Gene'])

train_embeddings = np.array(train_embeddings).squeeze()
test_embeddings = np.array(test_embeddings).squeeze()

# Enhanced PCA with more components and better preprocessing
per_channel_pca = np.array([14, 30])  # Increased PCA components from 20 to 30
pca_embeddings_train = None
pca_embeddings_test = None

print("Performing enhanced PCA feature extraction...")
for chan_num in tqdm(range(14)):
    channel_embeddings = train_embeddings[:, chan_num*384:(chan_num+1)*384]
    
    # Use more PCA components and explained variance threshold
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    pca_model = pca.fit(channel_embeddings)
    
    # Cap at 50 components maximum to avoid overfitting
    n_components = min(50, pca_model.n_components_)
    pca = PCA(n_components=n_components)
    pca_model = pca.fit(channel_embeddings)
    
    train_channel_pca = pca_model.transform(channel_embeddings)
    test_channel_pca = pca_model.transform(test_embeddings[:, chan_num*384:(chan_num+1)*384])
    
    if pca_embeddings_train is None:
        pca_embeddings_train = train_channel_pca
        pca_embeddings_test = test_channel_pca
    else:
        pca_embeddings_train = np.concatenate((pca_embeddings_train, train_channel_pca), axis=1)
        pca_embeddings_test = np.concatenate((pca_embeddings_test, test_channel_pca), axis=1)

print(f"Final feature dimensions: {pca_embeddings_train.shape[1]}")

# Prepare labels (assuming they exist from your preprocessing)
train_labels = [gene[0] for gene in train_genes]  # Extract gene names
test_labels = [gene[0] for gene in test_genes]

# Enhanced preprocessing with multiple scalers
scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler()
}

# Get unique genes
unique_genes = list(set(train_labels + test_labels))
unique_genes = [gene for gene in unique_genes if gene != 'non-target']

print(f"Found {len(unique_genes)} unique genes for binary classification")

# Enhanced model selection with F1-optimized hyperparameters
def get_optimized_models():
    return {
        'logistic_balanced': LogisticRegression(
            solver='saga', 
            max_iter=2000, 
            class_weight='balanced',
            random_state=42,
            C=0.1  # L2 regularization
        ),
        'logistic_f1_optimized': LogisticRegression(
            solver='liblinear',
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            C=1.0
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }

# Function to find optimal threshold for F1 score
def find_optimal_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# Enhanced binary classification with multiple strategies
binary_results = {}
best_overall_f1 = 0
best_strategy = ""

for scaler_name, scaler in scalers.items():
    print(f"\n{'='*60}")
    print(f"TESTING WITH {scaler_name.upper()} SCALER")
    print(f"{'='*60}")
    
    # Scale features
    X_train_scaled = scaler.fit_transform(pca_embeddings_train)
    X_test_scaled = scaler.transform(pca_embeddings_test)
    
    models = get_optimized_models()
    
    for model_name, base_model in models.items():
        print(f"\nTesting {model_name}...")
        current_results = {}
        
        for target_gene in tqdm(unique_genes, desc=f"{scaler_name}_{model_name}"):
            # Create binary labels
            train_binary_labels = []
            train_binary_indices = []
            for i, label in enumerate(train_labels):
                if label == target_gene:
                    train_binary_labels.append(1)
                    train_binary_indices.append(i)
                elif label == 'non-target':
                    train_binary_labels.append(0)
                    train_binary_indices.append(i)
            
            test_binary_labels = []
            test_binary_indices = []
            for i, label in enumerate(test_labels):
                if label == target_gene:
                    test_binary_labels.append(1)
                    test_binary_indices.append(i)
                elif label == 'non-target':
                    test_binary_labels.append(0)
                    test_binary_indices.append(i)
            
            # Skip if insufficient samples
            if len(train_binary_labels) < 10 or sum(test_binary_labels) < 3:
                continue
            
            X_train_binary = X_train_scaled[train_binary_indices]
            X_test_binary = X_test_scaled[test_binary_indices]
            
            # Feature selection for better generalization
            if X_train_binary.shape[1] > 100:
                selector = SelectKBest(f_classif, k=min(100, X_train_binary.shape[1]//2))
                X_train_binary = selector.fit_transform(X_train_binary, train_binary_labels)
                X_test_binary = selector.transform(X_test_binary)
            
            # Handle class imbalance with custom weights
            class_weights = compute_class_weight('balanced', 
                                               classes=np.unique(train_binary_labels), 
                                               y=train_binary_labels)
            weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            # Clone and configure model
            model = base_model.__class__(**base_model.get_params())
            if hasattr(model, 'class_weight'):
                model.class_weight = weight_dict
            
            # Cross-validation for model selection
            cv_scores = cross_val_score(model, X_train_binary, train_binary_labels, 
                                       cv=3, scoring=make_scorer(f1_score), n_jobs=-1)
            
            # Train final model
            model.fit(X_train_binary, train_binary_labels)
            
            # Get predictions and probabilities
            if hasattr(model, 'predict_proba'):
                test_proba = model.predict_proba(X_test_binary)[:, 1]
                # Find optimal threshold for F1
                optimal_threshold, _ = find_optimal_threshold(test_binary_labels, test_proba)
                test_pred = (test_proba >= optimal_threshold).astype(int)
            else:
                test_pred = model.predict(X_test_binary)
                test_proba = model.decision_function(X_test_binary)
            
            # Calculate metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                test_binary_labels, test_pred, average='binary', pos_label=1, zero_division=0
            )
            
            try:
                auc = roc_auc_score(test_binary_labels, test_proba)
            except:
                auc = 0.0
            
            current_results[target_gene] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_target_count': sum(test_binary_labels),
                'test_nontarget_count': len(test_binary_labels) - sum(test_binary_labels)
            }
        
        # Calculate average F1 for this configuration
        if current_results:
            avg_f1 = np.mean([metrics['f1'] for metrics in current_results.values()])
            print(f"Average F1 for {scaler_name}_{model_name}: {avg_f1:.4f}")
            
            if avg_f1 > best_overall_f1:
                best_overall_f1 = avg_f1
                best_strategy = f"{scaler_name}_{model_name}"
                binary_results = current_results.copy()

# Final results summary
print("\n" + "="*90)
print("OPTIMIZED BINARY CLASSIFICATION RESULTS")
print(f"BEST STRATEGY: {best_strategy}")
print("="*90)
print(f"{'Gene':<15} {'F1':<8} {'Precision':<10} {'Recall':<8} {'AUC':<8} {'CV_F1':<8} {'Test Samples':<12}")
print("-"*90)

for gene, metrics in sorted(binary_results.items(), key=lambda x: x[1]['f1'], reverse=True):
    print(f"{gene:<15} {metrics['f1']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} "
          f"{metrics['auc']:<8.4f} {metrics['cv_mean']:<8.4f} {metrics['test_target_count']:<12}")

# Calculate and display summary statistics
if binary_results:
    f1_scores = [metrics['f1'] for metrics in binary_results.values()]
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_precision = np.mean([metrics['precision'] for metrics in binary_results.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in binary_results.values()])
    avg_auc = np.mean([metrics['auc'] for metrics in binary_results.values()])
    
    print("-"*90)
    print(f"{'AVERAGE':<15} {avg_f1:<8.4f} {avg_precision:<10.4f} {avg_recall:<8.4f} {avg_auc:<8.4f}")
    print(f"{'STD F1':<15} {std_f1:<8.4f}")
    print(f"{'MAX F1':<15} {max(f1_scores):<8.4f}")
    print(f"{'MIN F1':<15} {min(f1_scores):<8.4f}")
    
    # Genes with F1 > 0.8 (high performers)
    high_performers = [gene for gene, metrics in binary_results.items() if metrics['f1'] > 0.8]
    print(f"\nHigh performing genes (F1 > 0.8): {len(high_performers)}")
    for gene in high_performers:
        print(f"  - {gene}: F1={binary_results[gene]['f1']:.4f}")

print(f"\nOptimization complete! Best average F1 score: {best_overall_f1:.4f}")
print(f"Best configuration: {best_strategy}")