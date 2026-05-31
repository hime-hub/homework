#!/usr/bin/env python
# coding: utf-8

"""
Transportation Mode Detection - Introduction to Machine Learning

This script performs supervised classification (Logistic Regression, Support Vector Machine, and Decision Tree)
and unsupervised learning (K-Means Clustering and PCA) on GPS trajectory features.
It loads 'traj_010_labeled_with_features.csv' and evaluates the models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import tree

def main():
    input_file = './traj_010_labeled_with_features.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Dataset '{input_file}' not found. Run 01_feature_extraction.py first.")
        return
        
    print(f"Loading feature dataset from {input_file}...")
    df_raw = pd.read_csv(input_file, index_col=0)
    
    # Create dataframe holding features per trip
    # In this script, we focus on the features' mean scores per trip
    df = df_raw.groupby('trans_trip').agg({
        'distance': 'mean',
        'speed': 'mean',
        'accel': 'mean',
        'angle': 'mean',
        'angular_velocity': 'mean',
        'trans_mode': 'first',
    })
    
    # Drop rows holding np.nan
    df.dropna(inplace=True)
    
    print("\nDataset Summary:")
    print(df.describe())
    
    print("\nTransportation Mode Counts:")
    print(df.trans_mode.value_counts())
    
    # Merge several labels and use three labels: vehicle (car/taxi/bus), walk, and train (train/subway)
    trans_mode_map = {
        'bus': 'vehicle',
        'car': 'vehicle',
        'taxi': 'vehicle',
        'subway': 'train',
        'walk': 'walk',
        'train': 'train'
    }
    df['trans_mode'] = df.trans_mode.map(trans_mode_map)
    
    print("\nTransportation Mode Counts after mapping:")
    print(df.trans_mode.value_counts())
    
    # Features (X) and Labels (y)
    X = df[['speed', 'accel', 'angular_velocity']]
    y = df['trans_mode']
    
    print("\nFeature Correlations:")
    print(X.corr())
    
    # Split into training and validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    
    # -------------------------------------------------------------
    # 1. Supervised Learning
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("1. Supervised Learning Models")
    print("="*50)
    
    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    lg_model = LogisticRegression(max_iter=1000)
    lg_model.fit(X_train, y_train)
    lg_y_predicted = lg_model.predict(X_test)
    print('Accuracy score (w/ training data): {:.4f}'.format(lg_model.score(X_train, y_train)))
    print('Accuracy score (w/ validation data): {:.4f}'.format(lg_model.score(X_test, y_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, lg_y_predicted))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, lg_y_predicted),
                       columns=['train', 'vehicle', 'walk'],
                       index=['train', 'vehicle', 'walk']))
                       
    # --- Support Vector Machine ---
    print("\n--- Support Vector Machine (SVM) ---")
    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    svm_y_predicted = svm_clf.predict(X_test)
    print("Accuracy score (w/ training data): {:.4f}".format(svm_clf.score(X_train, y_train)))
    print("Accuracy score (w/ validation data): {:.4f}".format(svm_clf.score(X_test, y_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, svm_y_predicted))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, svm_y_predicted),
                       columns=['train', 'vehicle', 'walk'],
                       index=['train', 'vehicle', 'walk']))
                       
    # --- Decision Tree Classifier ---
    print("\n--- Decision Tree Classifier ---")
    tree_clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    tree_clf.fit(X_train, y_train)
    tree_y_predicted = tree_clf.predict(X_test)
    print("Accuracy score (w/ training data): {:.4f}".format(tree_clf.score(X_train, y_train)))
    print("Accuracy score (w/ validation data): {:.4f}".format(tree_clf.score(X_test, y_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, tree_y_predicted))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, tree_y_predicted),
                       columns=['train', 'vehicle', 'walk'],
                       index=['train', 'vehicle', 'walk']))
                       
    # Visualize Decision Tree
    fig, ax = plt.subplots(figsize=(10, 6))
    tree.plot_tree(
        tree_clf, ax=ax, fontsize=8,
        feature_names=X.columns.tolist(),
        class_names=y.unique().tolist(),
        filled=True,
        rounded=True,
    )
    ax.set_title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig('plot_decision_tree.png')
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Unsupervised Learning
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("2. Unsupervised Learning Models")
    print("="*50)
    
    # --- K-Means Clustering ---
    print("\n--- K-Means Clustering (n_clusters=3) ---")
    X_kmeans = df[['angular_velocity', 'speed']]
    kmeans_model = KMeans(n_clusters=3, random_state=0)
    kmeans_model.fit(X_kmeans)
    y_km = kmeans_model.predict(X_kmeans)
    
    # Plot K-Means results vs Ground Truth
    y_colors = y.map({'train': 'navy', 'vehicle': 'turquoise', 'walk': 'darkorange'})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, ax in enumerate(axs):
        ax.scatter(df['angular_velocity'], df['speed'], c=[y_km, y_colors][i], alpha=0.8)
        ax.set_xlabel('angular_velocity')
        ax.set_ylabel('speed')
        ax.set_title(['K-Means Clustering', 'Ground Truth Class'][i])
    plt.tight_layout()
    plt.savefig('plot_kmeans_vs_truth.png')
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()
    
    # --- PCA Dimensionality Reduction ---
    print("\n--- Principal Component Analysis (PCA) ---")
    X_pca = df[['distance', 'speed', 'accel', 'angle', 'angular_velocity']]
    # Standardize features before running PCA
    X_pca_scaled = StandardScaler().fit_transform(X_pca)
    
    pca = PCA(n_components=2)
    X_pca_res = pca.fit_transform(X_pca_scaled)
    
    print("Explained variance ratio: {}".format(pca.explained_variance_ratio_))
    
    # Plot PCA results
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_pca = ['navy', 'turquoise', 'darkorange']
    target_names = y.unique()
    for color, target_name in zip(colors_pca, target_names):
        ax.scatter(X_pca_res[y == target_name, 0], X_pca_res[y == target_name, 1],
                   color=color, alpha=0.8, label=target_name)
    ax.legend(loc='best', shadow=False, scatterpoints=1, frameon=False)
    ax.set_title('PCA of GPS Dataset')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig('plot_pca.png')
    plt.show(block=False)
    plt.pause(2.0)
    plt.close()

if __name__ == '__main__':
    main()
