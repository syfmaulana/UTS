# Import semua library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.tree import plot_tree
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Set style untuk plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== ANALISIS KLASIFIKASI DATASET IRIS ===\n")

# =============================================================================
# LANGKAH 1: LOAD DAN EKSPLORASI DATA
# =============================================================================

print("1. LOADING DAN EKSPLORASI DATA")
print("-" * 40)

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# Eksplorasi dasar
print("Shape:", df.shape)
print("\nInfo dataset:")
print(df.info())
print("\n5 data pertama:")
print(df.head())
print("\nDeskripsi statistik:")
print(df.describe())
print("\nDistribusi kelas:")
print(df['species'].value_counts())

# =============================================================================
# LANGKAH 2: EDA (EXPLORATORY DATA ANALYSIS)
# =============================================================================

print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Visualisasi distribusi fitur per species
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for i, feature in enumerate(iris.feature_names):
    row, col = i // 2, i % 2
    sns.boxplot(x='species', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'Distribusi {feature} per Species', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('Species')
    axes[row, col].set_ylabel(feature)
plt.suptitle('DISTRIBUSI FITUR UNTUK SETIAP SPECIES', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Matrix korelasi
plt.figure(figsize=(10, 8))
correlation_matrix = df[iris.feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('MATRIX KORELASI FITUR', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Pairplot untuk melihat hubungan antar fitur
print("\nMembuat pairplot...")
sns.pairplot(df, hue='species', diag_kind='hist', palette='viridis')
plt.suptitle('PAIRPLOT: HUBUNGAN ANTAR FITUR', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# =============================================================================
# LANGKAH 3: PREPROCESSING DATA
# =============================================================================

print("\n3. PREPROCESSING DATA")
print("-" * 40)

# Pisahkan fitur dan target
X = df[iris.feature_names]
y = df['target']

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardisasi fitur (penting untuk SVM dan KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")
print(f"Jumlah kelas dalam training set: {np.bincount(y_train)}")
print(f"Jumlah kelas dalam testing set: {np.bincount(y_test)}")

# =============================================================================
# LANGKAH 4: IMPLEMENTASI MODEL KLASIFIKASI
# =============================================================================

print("\n4. IMPLEMENTASI MODEL KLASIFIKASI")
print("-" * 40)

# Dictionary untuk menyimpan model dan prediksi
models = {}
predictions = {}
prediction_probabilities = {}

# 4.1 Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)
models['Logistic Regression'] = lr_model
predictions['Logistic Regression'] = lr_pred
prediction_probabilities['Logistic Regression'] = lr_pred_proba
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")

# 4.2 Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_pred_proba = dt_model.predict_proba(X_test)
models['Decision Tree'] = dt_model
predictions['Decision Tree'] = dt_pred
prediction_probabilities['Decision Tree'] = dt_pred_proba
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")

# 4.3 K-Nearest Neighbors
print("Training K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_pred_proba = knn_model.predict_proba(X_test_scaled)
models['K-Nearest Neighbors'] = knn_model
predictions['K-Nearest Neighbors'] = knn_pred
prediction_probabilities['K-Nearest Neighbors'] = knn_pred_proba
print(f"KNN Accuracy: {accuracy_score(y_test, knn_pred):.4f}")

# 4.4 Support Vector Machine
print("Training Support Vector Machine...")
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_pred_proba = svm_model.predict_proba(X_test_scaled)
models['Support Vector Machine'] = svm_model
predictions['Support Vector Machine'] = svm_pred
prediction_probabilities['Support Vector Machine'] = svm_pred_proba
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")

# =============================================================================
# LANGKAH 5: EVALUASI MODEL
# =============================================================================

print("\n5. EVALUASI MODEL")
print("-" * 40)

# 5.1 Confusion Matrix
print("Membuat Confusion Matrix...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names, 
                ax=axes[idx], cbar=False)
    axes[idx].set_title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontweight='bold')
    axes[idx].set_ylabel('True Label', fontweight='bold')

plt.suptitle('CONFUSION MATRIX UNTUK SEMUA MODEL', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 5.2 Metrics Comparison
print("Menghitung metrik evaluasi...")

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Untuk multiclass, kita ambil weighted average
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# Evaluasi semua model
results = []
for name, pred in predictions.items():
    results.append(evaluate_model(y_test, pred, name))

# Buat DataFrame untuk perbandingan
results_df = pd.DataFrame(results)
print("\nPERBANDINGAN METRIK SEMUA MODEL:")
print("=" * 50)
print(results_df.round(4))
print("=" * 50)

# Visualisasi perbandingan metrics
metrics_plot = results_df.melt(id_vars=['Model'], 
                              value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                              var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_plot, palette='Set2')
plt.title('PERBANDINGAN METRIK EVALUASI MODEL', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 5.3 ROC Curve (One-vs-Rest untuk Multiclass)
print("Membuat ROC Curve...")

# Binarize output untuk multiclass ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Plot ROC Curve untuk setiap model
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.ravel()
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])

for idx, (model_name, y_score) in enumerate(prediction_probabilities.items()):
    # Hitung ROC curve dan ROC area untuk setiap class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    for i, color in zip(range(n_classes), colors):
        axes[idx].plot(fpr[i], tpr[i], color=color, lw=2,
                      label=f'Class {iris.target_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    axes[idx].plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    axes[idx].set_xlim([0.0, 1.0])
    axes[idx].set_ylim([0.0, 1.05])
    axes[idx].set_xlabel('False Positive Rate', fontweight='bold')
    axes[idx].set_ylabel('True Positive Rate', fontweight='bold')
    axes[idx].set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    axes[idx].legend(loc="lower right")
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('ROC CURVES UNTUK SEMUA MODEL (One-vs-Rest)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 5.4 Visualisasi Decision Tree
print("Visualisasi Decision Tree...")
plt.figure(figsize=(20, 12))
plot_tree(dt_model, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True, 
          rounded=True,
          fontsize=12,
          proportion=True)
plt.title('VISUALISASI DECISION TREE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# LANGKAH 6: ANALISIS DAN KESIMPULAN
# =============================================================================

print("\n6. ANALISIS DAN KESIMPULAN")
print("=" * 60)
print("HASIL AKHIR PERBANDINGAN MODEL KLASIFIKASI")
print("=" * 60)

# Urutkan berdasarkan accuracy terbaik
results_df_sorted = results_df.sort_values('Accuracy', ascending=False)
print(results_df_sorted.to_string(index=False))

# Model terbaik
best_model = results_df_sorted.iloc[0]
print(f"\n⭐ MODEL TERBAIK: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.4f}")
print(f"   Precision: {best_model['Precision']:.4f}")
print(f"   Recall: {best_model['Recall']:.4f}")
print(f"   F1-Score: {best_model['F1-Score']:.4f}")

print("\n" + "=" * 60)
print("KESIMPULAN DAN ANALISIS")
print("=" * 60)

print("""
1. PERFORMA TERBAIK:
Berdasarkan hasil evaluasi, model {} menunjukkan performa terbaik 
dengan akurasi {:.2f}% dan F1-Score {:.4f}.

2. PERBANDINGAN ALGORITMA:
- Logistic Regression: Model linear yang bekerja baik dengan data terstandardisasi
- Decision Tree: Mudah diinterpretasi namun rentan overfitting
- K-Nearest Neighbors: Berdasarkan similarity, sensitif terhadap scaling
- Support Vector Machine: Mencari hyperplane optimal, baik untuk data terpisah jelas

3. ANALISIS CONFUSION MATRIX:
Semua model menunjukkan performa yang sangat baik dalam memprediksi 
ketiga kelas (setosa, versicolor, virginica). Kesalahan klasifikasi 
minimal terjadi antara versicolor dan virginica yang memiliki 
karakteristik yang lebih mirip.

4. REKOMENDASI:
Untuk dataset Iris, {} direkomendasikan sebagai model terbaik. 
Namun semua model menunjukkan performa yang sangat baik (>95% accuracy) 
karena dataset yang well-structured dan terpisah dengan jelas.

5. POTENSI IMPROVEMENT:
- Tuning hyperparameter untuk optimasi lebih lanjut
- Cross-validation untuk evaluasi yang lebih robust
- Feature engineering jika diperlukan
""".format(best_model['Model'], best_model['Accuracy']*100, best_model['F1-Score'], best_model['Model']))

print("ANALISIS SELESAI! 🎯")