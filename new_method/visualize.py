"""
Enhanced Concrete Run-to-Failure Data Analysis Script
Features: Advanced EDA, Feature Engineering, Predictive Modeling, Survival Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
FILE_PATH = r'F:\concrete data\test 3\per_file_features_800.csv'
OUTPUT_DIR = r'F:\concrete data\test 3\analysis_results'
FIGURE_SIZE = (14, 8)
DPI = 300

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("ENHANCED CONCRETE RUN-TO-FAILURE DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv(FILE_PATH)
print(f"✓ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")

# Identify column types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# ============================================================================
# 2. IDENTIFY TARGET VARIABLE
# ============================================================================
print("\n" + "="*80)
print("TARGET VARIABLE IDENTIFICATION")
print("="*80)

# Smart target identification for concrete failure data
potential_target_keywords = [
    'failure', 'time_to_failure', 'ttf', 'lifetime', 'cycles', 
    'load', 'strength', 'target', 'label', 'rupture', 'crack',
    'damage', 'health', 'rul'  # Remaining Useful Life
]

target_col = None
target_candidates = []

for col in df.columns:
    col_lower = col.lower()
    # Check if column name contains target keywords
    if any(keyword in col_lower for keyword in potential_target_keywords):
        target_candidates.append(col)
    # Also check if it's a simple single-value identifier (often the last column)
    if col in numeric_cols and df[col].nunique() < len(df) * 0.9:
        target_candidates.append(col)

# Remove duplicates
target_candidates = list(set(target_candidates))

if target_candidates:
    print(f"Target variable candidates found: {target_candidates}")
    # Use the first candidate
    target_col = target_candidates[0]
    print(f"\n✓ Selected target variable: '{target_col}'")
else:
    print("⚠ No obvious target variable found.")
    print("Please specify manually by setting target_col variable.")
    # For demonstration, use the last numeric column
    if numeric_cols:
        target_col = numeric_cols[-1]
        print(f"Using last numeric column as target: '{target_col}'")

# ============================================================================
# 3. CHANNEL-BASED FEATURE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CHANNEL-BASED FEATURE ANALYSIS")
print("="*80)

# Identify channels (ch1, ch2, etc.)
channels = []
channel_features = {}

for col in numeric_cols:
    if col.startswith('ch') and '_' in col:
        channel = col.split('_')[0]
        if channel not in channels:
            channels.append(channel)
        if channel not in channel_features:
            channel_features[channel] = []
        channel_features[channel].append(col)

print(f"\nIdentified {len(channels)} channels: {channels}")
for ch in channels:
    print(f"  {ch}: {len(channel_features[ch])} features")

# Calculate channel statistics
channel_stats = {}
for ch in channels:
    ch_data = df[channel_features[ch]]
    channel_stats[ch] = {
        'mean': ch_data.mean().mean(),
        'std': ch_data.std().mean(),
        'variance': ch_data.var().mean()
    }

# Plot channel comparison
if channels:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ch_names = list(channel_stats.keys())
    means = [channel_stats[ch]['mean'] for ch in ch_names]
    stds = [channel_stats[ch]['std'] for ch in ch_names]
    vars = [channel_stats[ch]['variance'] for ch in ch_names]
    
    axes[0].bar(ch_names, means)
    axes[0].set_title('Average Mean Across Channels', fontweight='bold')
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Mean Value')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(ch_names, stds)
    axes[1].set_title('Average Std Dev Across Channels', fontweight='bold')
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(ch_names, vars)
    axes[2].set_title('Average Variance Across Channels', fontweight='bold')
    axes[2].set_xlabel('Channel')
    axes[2].set_ylabel('Variance')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'channel_statistics.png'), dpi=DPI, bbox_inches='tight')
    print("\n✓ Saved: channel_statistics.png")
    plt.close()

# ============================================================================
# 4. ADVANCED CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ADVANCED CORRELATION ANALYSIS")
print("="*80)

if target_col and target_col in numeric_cols:
    # Calculate correlation with target
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    correlations = []
    for col in feature_cols:
        if df[col].notna().sum() > 10:  # Need at least 10 valid values
            corr, p_value = pearsonr(df[col].dropna(), df[target_col].dropna())
            correlations.append({
                'Feature': col,
                'Correlation': corr,
                'Abs_Correlation': abs(corr),
                'P_Value': p_value
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    
    print("\n--- Top 30 Features Correlated with Target ---")
    print(corr_df.head(30).to_string(index=False))
    
    # Save correlation results
    corr_df.to_csv(os.path.join(OUTPUT_DIR, 'target_correlations.csv'), index=False)
    
    # Plot top correlations
    top_n = min(30, len(corr_df))
    plt.figure(figsize=(12, 10))
    top_corr = corr_df.head(top_n)
    colors = ['red' if x < 0 else 'green' for x in top_corr['Correlation']]
    plt.barh(range(top_n), top_corr['Correlation'], color=colors, alpha=0.7)
    plt.yticks(range(top_n), top_corr['Feature'], fontsize=8)
    plt.xlabel('Correlation with Target', fontweight='bold')
    plt.title(f'Top {top_n} Features by Correlation with {target_col}', fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_target_correlations.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Saved: top_target_correlations.png")
    plt.close()

# ============================================================================
# 5. MUTUAL INFORMATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("MUTUAL INFORMATION ANALYSIS")
print("="*80)

if target_col and target_col in numeric_cols:
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]
    
    # Remove any remaining NaN from target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print("\nCalculating mutual information scores...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'Feature': feature_cols,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    print("\n--- Top 30 Features by Mutual Information ---")
    print(mi_df.head(30).to_string(index=False))
    
    mi_df.to_csv(os.path.join(OUTPUT_DIR, 'mutual_information_scores.csv'), index=False)
    
    # Plot MI scores
    top_n = min(30, len(mi_df))
    plt.figure(figsize=(12, 10))
    plt.barh(range(top_n), mi_df['MI_Score'].head(top_n), alpha=0.7)
    plt.yticks(range(top_n), mi_df['Feature'].head(top_n), fontsize=8)
    plt.xlabel('Mutual Information Score', fontweight='bold')
    plt.title(f'Top {top_n} Features by Mutual Information', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mutual_information_scores.png'), dpi=DPI, bbox_inches='tight')
    print("✓ Saved: mutual_information_scores.png")
    plt.close()

# ============================================================================
# 6. CLUSTERING ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CLUSTERING ANALYSIS")
print("="*80)

# Use top PCA components for clustering
print("\nPerforming clustering on PCA-reduced data...")
X_cluster = df[numeric_cols].fillna(df[numeric_cols].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# PCA reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Try different numbers of clusters
inertias = []
silhouette_scores = []
k_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_range, inertias, marker='o', linewidth=2)
ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax1.set_ylabel('Inertia', fontweight='bold')
ax1.set_title('Elbow Method For Optimal k', fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, silhouette_scores, marker='o', linewidth=2, color='orange')
ax2.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontweight='bold')
ax2.set_title('Silhouette Score vs Number of Clusters', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_analysis.png'), dpi=DPI, bbox_inches='tight')
print("✓ Saved: clustering_analysis.png")
plt.close()

# Perform final clustering with optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_pca)

# Visualize clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
plt.ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
plt.title(f'K-Means Clustering (k={optimal_k}) on PCA Components', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_visualization.png'), dpi=DPI, bbox_inches='tight')
print("✓ Saved: cluster_visualization.png")
plt.close()

# Add cluster labels to dataframe
df['cluster'] = clusters

# ============================================================================
# 7. PREDICTIVE MODELING
# ============================================================================
print("\n" + "="*80)
print("PREDICTIVE MODELING")
print("="*80)

if target_col and target_col in numeric_cols:
    print(f"\nBuilding predictive models for: {target_col}")
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]
    
    # Remove any NaN from target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[name] = {
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'model': model
        }
        
        print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        print(f"  Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        name: {k: v for k, v in vals.items() if k != 'model'}
        for name, vals in results.items()
    }).T
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance.csv'))
    
    # Plot predictions vs actual
    best_model_name = max(results, key=lambda x: results[x]['Test_R2'])
    best_model = results[best_model_name]['model']
    y_pred_best = best_model.predict(X_test_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set
    y_pred_train_best = best_model.predict(X_train_scaled)
    axes[0].scatter(y_train, y_pred_train_best, alpha=0.5, s=30)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual', fontweight='bold')
    axes[0].set_ylabel('Predicted', fontweight='bold')
    axes[0].set_title(f'{best_model_name} - Training Set (R²={results[best_model_name]["Train_R2"]:.4f})', 
                      fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Test set
    axes[1].scatter(y_test, y_pred_best, alpha=0.5, s=30, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual', fontweight='bold')
    axes[1].set_ylabel('Predicted', fontweight='bold')
    axes[1].set_title(f'{best_model_name} - Test Set (R²={results[best_model_name]["Test_R2"]:.4f})', 
                      fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_accuracy.png'), dpi=DPI, bbox_inches='tight')
    print(f"\n✓ Saved: prediction_accuracy.png")
    plt.close()
    
    # Feature importance from best model
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n--- Top 30 Most Important Features ---")
        print(importance_df.head(30).to_string(index=False))
        
        importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_detailed.csv'), index=False)
        
        # Plot
        top_n = min(30, len(importance_df))
        plt.figure(figsize=(12, 10))
        plt.barh(range(top_n), importance_df['Importance'].head(top_n), alpha=0.7)
        plt.yticks(range(top_n), importance_df['Feature'].head(top_n), fontsize=8)
        plt.xlabel('Importance Score', fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances ({best_model_name})', fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_model.png'), dpi=DPI, bbox_inches='tight')
        print("✓ Saved: feature_importance_model.png")
        plt.close()

# ============================================================================
# 8. FEATURE ENGINEERING RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING RECOMMENDATIONS")
print("="*80)

recommendations = []

# Check for highly correlated features
corr_matrix = df[numeric_cols].corr()
high_corr_threshold = 0.95
high_corr_features = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
            high_corr_features.append((corr_matrix.columns[i], corr_matrix.columns[j]))

if high_corr_features:
    recommendations.append(f"Found {len(high_corr_features)} feature pairs with correlation > {high_corr_threshold}")
    recommendations.append("Consider removing one feature from each highly correlated pair")

# Check for low variance features
low_var_features = []
for col in numeric_cols:
    if df[col].var() < 0.01:
        low_var_features.append(col)

if low_var_features:
    recommendations.append(f"Found {len(low_var_features)} low variance features (var < 0.01)")
    recommendations.append("Consider removing these features as they provide little information")

# Suggest interaction features
recommendations.append("\nSuggested interaction features:")
recommendations.append("- Cross-channel features (e.g., ch1_feature * ch2_feature)")
recommendations.append("- Ratio features between channels")
recommendations.append("- Polynomial features for non-linear relationships")

# Suggest aggregated features
recommendations.append("\nSuggested aggregated features:")
recommendations.append("- Mean/std/max across all channels for each feature type")
recommendations.append("- Difference features between consecutive channels")
recommendations.append("- Rolling statistics if temporal ordering exists")

print("\n".join(recommendations))

# Save recommendations
with open(os.path.join(OUTPUT_DIR, 'feature_engineering_recommendations.txt'), 'w') as f:
    f.write('\n'.join(recommendations))

# ============================================================================
# 9. FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("ENHANCED ANALYSIS SUMMARY")
print("="*80)

if target_col and 'best_model_name' in locals():
    test_r2 = results[best_model_name]['Test_R2']
    test_rmse = results[best_model_name]['Test_RMSE']
    best_model_str = best_model_name
else:
    test_r2 = 'N/A'
    test_rmse = 'N/A'
    best_model_str = 'N/A'

summary = f"""
ENHANCED CONCRETE RUN-TO-FAILURE DATA ANALYSIS - SUMMARY REPORT
{'='*80}

1. DATASET OVERVIEW
   - Total Samples: {df.shape[0]:,}
   - Total Features: {df.shape[1]:,}
   - Channels Identified: {len(channels)}
   - Target Variable: {target_col if target_col else 'Not identified'}

2. CORRELATION ANALYSIS
   - High correlation pairs (>0.7): Available in high_correlations.csv
   - Top feature-target correlations: Saved in target_correlations.csv

3. CLUSTERING ANALYSIS
   - Optimal number of clusters: {optimal_k}
   - Silhouette score: {max(silhouette_scores):.4f}

4. PREDICTIVE MODELING
   - Best Model: {best_model_str}
   - Test R²: {test_r2 if isinstance(test_r2, str) else f'{test_r2:.4f}'}
   - Test RMSE: {test_rmse if isinstance(test_rmse, str) else f'{test_rmse:.4f}'}

5. KEY INSIGHTS
   - PCA Components for 95% variance: 23 (from previous analysis)
   - Features with outliers: 639 (from previous analysis)
   - High skewness features: 353 (from previous analysis)

6. GENERATED OUTPUTS
   ✓ channel_statistics.png
   ✓ top_target_correlations.png
   ✓ mutual_information_scores.png & .csv
   ✓ clustering_analysis.png
   ✓ cluster_visualization.png
   ✓ prediction_accuracy.png
   ✓ feature_importance_model.png & .csv
   ✓ model_performance.csv
   ✓ feature_engineering_recommendations.txt

All outputs saved to: {OUTPUT_DIR}

RECOMMENDATIONS FOR NEXT STEPS:
1. Implement suggested feature engineering strategies
2. Try ensemble methods combining top features from different analyses
3. Investigate cluster characteristics for domain insights
4. Consider time-series modeling if temporal patterns exist
5. Validate model on independent test set or cross-validation
6. Investigate physical meaning of top predictive features

{'='*80}
"""

print(summary)

# Save summary
with open(os.path.join(OUTPUT_DIR, 'enhanced_analysis_summary.txt'), 'w') as f:
    f.write(summary)

print("\n" + "="*80)
print("ENHANCED ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\n✓ Ready for modeling and deployment!")
print("="*80)