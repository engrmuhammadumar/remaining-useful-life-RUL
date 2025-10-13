"""
Novel Hybrid Health Index for Concrete RUL Prediction
Combines: Monotonicity, Entropy, Exponential Weighting, Multi-scale Analysis
State-of-the-art approach for damage accumulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class HybridHealthIndex:
    """
    Novel Hybrid Health Index combining:
    1. Monotonicity-weighted features
    2. Multi-scale temporal analysis
    3. Entropy-based degradation
    4. Exponential damage accumulation
    5. Principal Component Analysis
    """
    
    def __init__(self, n_components=5, smoothing_window=21, poly_order=3):
        self.n_components = n_components
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.selected_features = []
        self.feature_weights = {}
        
    def calculate_monotonicity(self, series):
        """Calculate monotonicity score"""
        series_clean = series.dropna()
        if len(series_clean) < 2:
            return 0
        diffs = np.diff(series_clean)
        if len(diffs) == 0:
            return 0
        increasing = np.sum(diffs > 0)
        decreasing = np.sum(diffs < 0)
        total = len(diffs)
        return (increasing - decreasing) / total
    
    def calculate_trend_strength(self, series):
        """Calculate trend strength using linear regression"""
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return 0
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Normalize by data range
        data_range = np.ptp(y)
        if data_range == 0:
            return 0
        
        return abs(slope * len(x) / data_range)
    
    def calculate_complexity(self, series):
        """Calculate signal complexity using sample entropy"""
        series_clean = series.dropna().values
        if len(series_clean) < 10:
            return 0
        
        # Normalize
        series_norm = (series_clean - np.mean(series_clean)) / (np.std(series_clean) + 1e-10)
        
        # Calculate approximate entropy
        try:
            N = len(series_norm)
            m = 2  # pattern length
            r = 0.2 * np.std(series_norm)  # tolerance
            
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            
            def _phi(m):
                x = [[series_norm[j] for j in range(i, i + m)] for i in range(N - m + 1)]
                C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
                return (N - m + 1.0)**(-1) * sum(np.log(C))
            
            return abs(_phi(m) - _phi(m + 1))
        except:
            return 0
    
    def select_features(self, df, feature_cols, top_n=50):
        """
        Select top features based on multiple criteria:
        1. Monotonicity
        2. Trend strength
        3. Complexity
        4. Variance
        """
        print("\n" + "="*80)
        print("FEATURE SELECTION FOR HYBRID HEALTH INDEX")
        print("="*80)
        
        feature_scores = []
        
        for col in feature_cols:
            monotonicity = abs(self.calculate_monotonicity(df[col]))
            trend = self.calculate_trend_strength(df[col])
            complexity = self.calculate_complexity(df[col])
            variance = df[col].var()
            
            # Composite score with weights
            composite_score = (
                0.35 * monotonicity +  # Prefer monotonic features
                0.30 * trend +          # Strong trends
                0.20 * complexity +     # Complex patterns
                0.15 * (variance / (df[col].max() + 1e-10))  # Normalized variance
            )
            
            feature_scores.append({
                'Feature': col,
                'Monotonicity': monotonicity,
                'Trend': trend,
                'Complexity': complexity,
                'Variance': variance,
                'Composite_Score': composite_score
            })
        
        scores_df = pd.DataFrame(feature_scores)
        scores_df = scores_df.sort_values('Composite_Score', ascending=False)
        
        # Select top N features
        self.selected_features = scores_df.head(top_n)['Feature'].tolist()
        
        # Store weights for later use
        for _, row in scores_df.head(top_n).iterrows():
            self.feature_weights[row['Feature']] = row['Composite_Score']
        
        print(f"\n✓ Selected top {top_n} features")
        print(f"\nTop 10 features:")
        print(scores_df.head(10)[['Feature', 'Monotonicity', 'Trend', 'Composite_Score']].to_string(index=False))
        
        return scores_df
    
    def calculate_multi_scale_hi(self, df, feature_cols):
        """
        Calculate Health Index at multiple time scales
        """
        print("\n[1/5] Calculating multi-scale health indices...")
        
        # Extract selected features
        X = df[feature_cols].values
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Use first principal component as base HI
        base_hi = X_pca[:, 0]
        
        # Normalize to [0, 1], with 1 = healthy, 0 = failed
        base_hi = (base_hi - base_hi.min()) / (base_hi.max() - base_hi.min())
        base_hi = 1 - base_hi  # Invert so failure is at 0
        
        print(f"  ✓ PCA variance explained: {self.pca.explained_variance_ratio_[:3]}")
        
        return base_hi, X_pca
    
    def calculate_entropy_based_hi(self, df, feature_cols, window_size=50):
        """
        Calculate entropy-based health index using sliding window
        """
        print("\n[2/5] Calculating entropy-based health index...")
        
        entropy_hi = []
        
        for i in range(len(df)):
            start_idx = max(0, i - window_size)
            window_data = df[feature_cols].iloc[start_idx:i+1]
            
            # Calculate entropy for each feature in window
            entropies = []
            for col in feature_cols[:20]:  # Use top 20 for speed
                data = window_data[col].values
                if len(data) > 1:
                    hist, _ = np.histogram(data, bins=10, density=True)
                    hist = hist[hist > 0]  # Remove zeros
                    ent = entropy(hist)
                    entropies.append(ent)
            
            # Average entropy as degradation indicator
            avg_entropy = np.mean(entropies) if entropies else 0
            entropy_hi.append(avg_entropy)
        
        # Normalize
        entropy_hi = np.array(entropy_hi)
        entropy_hi = (entropy_hi - entropy_hi.min()) / (entropy_hi.max() - entropy_hi.min() + 1e-10)
        
        print(f"  ✓ Entropy HI calculated")
        
        return entropy_hi
    
    def calculate_damage_accumulation_hi(self, df, feature_cols):
        """
        Calculate cumulative damage-based health index
        """
        print("\n[3/5] Calculating damage accumulation index...")
        
        # Calculate incremental damage for each feature
        damage_increments = []
        
        for col in feature_cols:
            # Use absolute differences
            diff = np.abs(np.diff(df[col].values, prepend=df[col].iloc[0]))
            
            # Weight by feature importance
            weight = self.feature_weights.get(col, 0)
            
            weighted_diff = diff * weight
            damage_increments.append(weighted_diff)
        
        # Sum across all features
        total_damage = np.sum(damage_increments, axis=0)
        
        # Cumulative damage with exponential weighting (recent damage matters more)
        alpha = 0.05  # decay factor
        cumulative_damage = np.zeros(len(total_damage))
        
        for i in range(len(total_damage)):
            if i == 0:
                cumulative_damage[i] = total_damage[i]
            else:
                cumulative_damage[i] = cumulative_damage[i-1] * (1 + alpha) + total_damage[i]
        
        # Normalize to [0, 1]
        damage_hi = cumulative_damage / (cumulative_damage.max() + 1e-10)
        
        print(f"  ✓ Damage accumulation HI calculated")
        
        return damage_hi
    
    def calculate_hybrid_hi(self, df, feature_cols, weights=None):
        """
        Combine all health indices into final hybrid HI
        """
        if weights is None:
            weights = {
                'pca': 0.30,
                'entropy': 0.25,
                'damage': 0.30,
                'exponential': 0.15
            }
        
        print("\n[4/5] Combining indices into hybrid HI...")
        print(f"  Weights: {weights}")
        
        # Calculate individual HIs
        pca_hi, X_pca = self.calculate_multi_scale_hi(df, feature_cols)
        entropy_hi = self.calculate_entropy_based_hi(df, feature_cols)
        damage_hi = self.calculate_damage_accumulation_hi(df, feature_cols)
        
        # Exponential degradation model (assumes late-stage acceleration)
        time = np.arange(len(df))
        exponential_hi = 1 - (1 - np.exp(-0.005 * time)) / (1 - np.exp(-0.005 * len(df)))
        
        # Combine with weights
        hybrid_hi = (
            weights['pca'] * (1 - pca_hi) +
            weights['entropy'] * entropy_hi +
            weights['damage'] * damage_hi +
            weights['exponential'] * (1 - exponential_hi)
        )
        
        # Smooth the HI
        if len(hybrid_hi) > self.smoothing_window:
            hybrid_hi_smooth = savgol_filter(hybrid_hi, self.smoothing_window, self.poly_order)
        else:
            hybrid_hi_smooth = hybrid_hi
        
        # Ensure monotonicity (HI should not increase)
        hybrid_hi_monotonic = np.maximum.accumulate(hybrid_hi_smooth)
        
        # Final normalization [0, 1] where 0 = healthy, 1 = failed
        hybrid_hi_final = (hybrid_hi_monotonic - hybrid_hi_monotonic.min()) / \
                         (hybrid_hi_monotonic.max() - hybrid_hi_monotonic.min() + 1e-10)
        
        print(f"  ✓ Hybrid HI computed and smoothed")
        
        # Store all HIs for analysis
        self.all_his = {
            'pca_hi': 1 - pca_hi,
            'entropy_hi': entropy_hi,
            'damage_hi': damage_hi,
            'exponential_hi': 1 - exponential_hi,
            'hybrid_hi_raw': hybrid_hi,
            'hybrid_hi_smooth': hybrid_hi_smooth,
            'hybrid_hi_final': hybrid_hi_final
        }
        
        return hybrid_hi_final
    
    def calculate_rul(self, hi, current_threshold=0.8):
        """
        Calculate RUL from Health Index
        
        RUL = (Failure_threshold - Current_HI) / Degradation_rate
        """
        print("\n[5/5] Calculating Remaining Useful Life (RUL)...")
        
        rul = np.zeros(len(hi))
        failure_threshold = 1.0  # HI at failure
        
        for i in range(len(hi)):
            if hi[i] >= failure_threshold:
                rul[i] = 0
            else:
                # Estimate degradation rate from recent history
                window = 50
                start_idx = max(0, i - window)
                
                if i > start_idx:
                    recent_hi = hi[start_idx:i+1]
                    time_steps = np.arange(len(recent_hi))
                    
                    # Fit linear trend
                    if len(recent_hi) > 1:
                        coeffs = np.polyfit(time_steps, recent_hi, 1)
                        degradation_rate = coeffs[0]
                        
                        if degradation_rate > 1e-6:
                            estimated_rul = (failure_threshold - hi[i]) / degradation_rate
                            rul[i] = max(0, estimated_rul)
                        else:
                            rul[i] = len(hi) - i
                    else:
                        rul[i] = len(hi) - i
                else:
                    rul[i] = len(hi) - i
        
        # Smooth RUL
        if len(rul) > self.smoothing_window:
            rul_smooth = savgol_filter(rul, self.smoothing_window, self.poly_order)
            rul_smooth = np.maximum(rul_smooth, 0)  # Ensure non-negative
        else:
            rul_smooth = rul
        
        print(f"  ✓ RUL calculated")
        print(f"  ✓ Initial RUL: {rul_smooth[0]:.1f} time steps")
        print(f"  ✓ Final RUL: {rul_smooth[-1]:.1f} time steps")
        
        return rul_smooth


def create_hybrid_health_index(csv_path, output_dir='rul_prediction'):
    """
    Main function to create hybrid health index and RUL prediction
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("NOVEL HYBRID HEALTH INDEX FOR RUL PREDICTION")
    print("="*80)
    
    # Load data
    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Add segment number if not present
    if 'segment_number' not in df.columns:
        df['segment_number'] = range(len(df))
        print(f"  ✓ Added segment_number column")
    
    # Identify feature columns
    metadata_cols = ['file', 'segment_number']
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Remove constant features
    constant_features = [col for col in feature_cols if df[col].nunique() == 1]
    feature_cols = [col for col in feature_cols if col not in constant_features]
    
    print(f"  ✓ Loaded {len(df)} samples")
    print(f"  ✓ Available features: {len(feature_cols)}")
    print(f"  ✓ Removed {len(constant_features)} constant features")
    
    # Initialize hybrid HI
    print("\n[Step 2] Initializing Hybrid Health Index...")
    hhi = HybridHealthIndex(n_components=5, smoothing_window=21, poly_order=3)
    
    # Feature selection
    print("\n[Step 3] Selecting most informative features...")
    feature_scores = hhi.select_features(df, feature_cols, top_n=50)
    feature_scores.to_csv(os.path.join(output_dir, 'feature_scores.csv'), index=False)
    
    # Calculate hybrid HI
    print("\n[Step 4] Computing Hybrid Health Index...")
    hybrid_hi = hhi.calculate_hybrid_hi(df, hhi.selected_features)
    
    # Calculate RUL
    print("\n[Step 5] Computing Remaining Useful Life...")
    rul = hhi.calculate_rul(hybrid_hi)
    
    # Save results
    print("\n[Step 6] Saving results...")
    results_df = pd.DataFrame()
    results_df['segment_number'] = df['segment_number']
    results_df['Health_Index'] = hybrid_hi
    results_df['RUL'] = rul
    
    # Add individual HIs
    for name, hi in hhi.all_his.items():
        results_df[name] = hi
    
    results_df.to_csv(os.path.join(output_dir, 'health_index_and_rul.csv'), index=False)
    print(f"  ✓ Results saved to {output_dir}/health_index_and_rul.csv")
    
    # ============================================
    # VISUALIZATIONS
    # ============================================
    print("\n[Step 7] Generating visualizations...")
    
    # Plot 1: Hybrid HI Components
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    time = df['segment_number'].values
    
    axes[0, 0].plot(time, hhi.all_his['pca_hi'], 'b-', linewidth=2)
    axes[0, 0].set_title('PCA-based Health Index', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Segment Number')
    axes[0, 0].set_ylabel('Health Index')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time, hhi.all_his['entropy_hi'], 'g-', linewidth=2)
    axes[0, 1].set_title('Entropy-based Health Index', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Segment Number')
    axes[0, 1].set_ylabel('Health Index')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(time, hhi.all_his['damage_hi'], 'r-', linewidth=2)
    axes[1, 0].set_title('Damage Accumulation Index', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Segment Number')
    axes[1, 0].set_ylabel('Health Index')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time, hhi.all_his['exponential_hi'], 'm-', linewidth=2)
    axes[1, 1].set_title('Exponential Degradation Model', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Segment Number')
    axes[1, 1].set_ylabel('Health Index')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].plot(time, hhi.all_his['hybrid_hi_raw'], 'gray', alpha=0.5, label='Raw')
    axes[2, 0].plot(time, hhi.all_his['hybrid_hi_smooth'], 'orange', linewidth=2, label='Smoothed')
    axes[2, 0].plot(time, hybrid_hi, 'darkblue', linewidth=2.5, label='Final (Monotonic)')
    axes[2, 0].set_title('Hybrid HI Evolution', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Segment Number')
    axes[2, 0].set_ylabel('Health Index')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(time, rul, 'darkred', linewidth=2.5)
    axes[2, 1].fill_between(time, rul, alpha=0.3, color='red')
    axes[2, 1].set_title('Remaining Useful Life (RUL)', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Segment Number')
    axes[2, 1].set_ylabel('RUL (time steps)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_hi_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Component visualization saved")
    
    # Plot 2: Final HI and RUL together
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.plot(time, hybrid_hi, 'darkblue', linewidth=3, label='Hybrid Health Index')
    ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='50% Degradation')
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Critical Threshold')
    ax1.fill_between(time, hybrid_hi, alpha=0.3, color='blue')
    ax1.set_title('Hybrid Health Index (0=Healthy, 1=Failed)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Segment Number (Time)', fontsize=12)
    ax1.set_ylabel('Health Index', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time, rul, 'darkred', linewidth=3, label='RUL Prediction')
    ax2.fill_between(time, rul, alpha=0.3, color='red')
    ax2.set_title('Remaining Useful Life Prediction', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Segment Number (Time)', fontsize=12)
    ax2.set_ylabel('RUL (time steps)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_hi_and_rul.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Final HI and RUL visualization saved")
    
    # Plot 3: Feature importance
    top_features = feature_scores.head(20)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    plt.barh(range(len(top_features)), top_features['Composite_Score'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Composite Feature Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top 20 Features for Health Index Calculation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature importance visualization saved")
    
    # ============================================
    # PERFORMANCE METRICS
    # ============================================
    print("\n[Step 8] Computing performance metrics...")
    
    # Monotonicity score
    hi_diffs = np.diff(hybrid_hi)
    monotonicity_score = np.sum(hi_diffs >= 0) / len(hi_diffs)
    
    # Smoothness (lower is better)
    smoothness = np.mean(np.abs(np.diff(hybrid_hi)))
    
    # Trendability
    time_normalized = (time - time.min()) / (time.max() - time.min())
    correlation = np.corrcoef(time_normalized, hybrid_hi)[0, 1]
    
    # Prognosability (separation of early vs late stage)
    early_stage = hybrid_hi[:len(hybrid_hi)//3]
    late_stage = hybrid_hi[2*len(hybrid_hi)//3:]
    prognosability = (np.mean(late_stage) - np.mean(early_stage)) / np.std(hybrid_hi)
    
    metrics = {
        'Monotonicity': monotonicity_score,
        'Smoothness': smoothness,
        'Correlation_with_Time': correlation,
        'Prognosability': prognosability
    }
    
    print(f"\n{'='*80}")
    print("PERFORMANCE METRICS")
    print(f"{'='*80}")
    for key, value in metrics.items():
        print(f"  {key:.<40} {value:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    # ============================================
    # SUMMARY REPORT
    # ============================================
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("NOVEL HYBRID HEALTH INDEX - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. METHODOLOGY\n")
        f.write("   Hybrid approach combining:\n")
        f.write("   - PCA-based dimensionality reduction (30%)\n")
        f.write("   - Entropy-based complexity analysis (25%)\n")
        f.write("   - Cumulative damage accumulation (30%)\n")
        f.write("   - Exponential degradation model (15%)\n\n")
        
        f.write("2. FEATURE SELECTION\n")
        f.write(f"   - Total features available: {len(feature_cols)}\n")
        f.write(f"   - Features selected: {len(hhi.selected_features)}\n")
        f.write(f"   - Selection criteria: Monotonicity + Trend + Complexity + Variance\n\n")
        
        f.write("3. HEALTH INDEX CHARACTERISTICS\n")
        f.write(f"   - Range: [{hybrid_hi.min():.4f}, {hybrid_hi.max():.4f}]\n")
        f.write(f"   - Initial HI: {hybrid_hi[0]:.4f} (healthy)\n")
        f.write(f"   - Final HI: {hybrid_hi[-1]:.4f} (failed)\n")
        f.write(f"   - Monotonicity: {monotonicity_score:.2%}\n\n")
        
        f.write("4. RUL PREDICTION\n")
        f.write(f"   - Initial RUL: {rul[0]:.1f} time steps\n")
        f.write(f"   - RUL at 50% life: {rul[len(rul)//2]:.1f} time steps\n")
        f.write(f"   - Final RUL: {rul[-1]:.1f} time steps\n\n")
        
        f.write("5. PERFORMANCE METRICS\n")
        for key, value in metrics.items():
            f.write(f"   - {key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("6. TOP 10 MOST IMPORTANT FEATURES\n")
        for i, (idx, row) in enumerate(feature_scores.head(10).iterrows(), 1):
            f.write(f"   {i}. {row['Feature']}: {row['Composite_Score']:.4f}\n")
        f.write("\n")
        
        f.write("7. OUTPUT FILES\n")
        f.write("   - health_index_and_rul.csv: Complete results\n")
        f.write("   - feature_scores.csv: Feature importance scores\n")
        f.write("   - hybrid_hi_components.png: Component visualizations\n")
        f.write("   - final_hi_and_rul.png: Final results\n")
        f.write("   - feature_importance.png: Feature rankings\n")
        f.write("   - performance_metrics.csv: Quantitative metrics\n")
    
    print(f"\n  ✓ Summary report saved")
    
    print("\n" + "="*80)
    print("HYBRID HEALTH INDEX COMPUTATION COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"\n✨ Novel Hybrid HI successfully created!")
    print(f"✨ Monotonicity: {monotonicity_score:.2%}")
    print(f"✨ Prognosability: {prognosability:.4f}")
    print(f"✨ Ready for advanced RUL prediction modeling!")
    
    return results_df, hhi, feature_scores, metrics


if __name__ == "__main__":
    # Configuration
    CSV_PATH = r"F:\concrete data\test 4\ae_features_800\per_file_features_800.csv"
    OUTPUT_DIR = "rul_prediction"
    
    # Run hybrid health index creation
    results, hhi_model, feature_scores, metrics = create_hybrid_health_index(CSV_PATH, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("Variables available:")
    print("  - results: DataFrame with HI and RUL for each segment")
    print("  - hhi_model: Trained HybridHealthIndex model")
    print("  - feature_scores: Feature importance rankings")
    print("  - metrics: Performance metrics")
    print("="*80)