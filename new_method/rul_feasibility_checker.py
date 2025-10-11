"""
RUL (Remaining Useful Life) Prediction Feasibility Analysis
Checks if your concrete dataset is suitable for RUL estimation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

FILE_PATH = r'F:\concrete data\test 3\per_file_features_800.csv'
OUTPUT_DIR = r'F:\concrete data\test 3\rul_analysis'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("RUL PREDICTION FEASIBILITY ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv(FILE_PATH)
print(f"\nLoaded: {df.shape[0]} rows x {df.shape[1]} columns")

# Get filename column
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if not categorical_cols:
    print("\n⚠️ WARNING: No filename column found!")
    filename_col = None
else:
    filename_col = categorical_cols[0]
    print(f"\nFilename column: '{filename_col}'")

# ============================================================================
# 1. EXAMINE FILENAME STRUCTURE
# ============================================================================
print("\n" + "="*80)
print("STEP 1: FILENAME STRUCTURE ANALYSIS")
print("="*80)

if filename_col:
    print("\nFirst 30 filenames:")
    for i, fname in enumerate(df[filename_col].head(30), 1):
        print(f"  {i:3d}. {fname}")
    
    print("\nLast 30 filenames:")
    for i, fname in enumerate(df[filename_col].tail(30), 1):
        print(f"  {i:3d}. {fname}")
    
    # Check for patterns
    print("\n--- Pattern Detection ---")
    
    # Pattern 1: Sequential numbers (file_001, file_002, ...)
    sequential = df[filename_col].str.extract(r'(\d{3,})', expand=False)
    if sequential.notna().sum() > 0:
        seq_nums = pd.to_numeric(sequential, errors='coerce')
        print(f"\n✓ Sequential Pattern Found:")
        print(f"  Range: [{seq_nums.min():.0f}, {seq_nums.max():.0f}]")
        print(f"  Unique: {seq_nums.nunique()}")
        print(f"  Sorted: {(seq_nums == seq_nums.sort_values()).all()}")
        
        # Check if monotonically increasing
        is_monotonic = (seq_nums.diff().dropna() > 0).all()
        print(f"  Monotonic: {is_monotonic}")
        
        if is_monotonic or seq_nums.is_monotonic_increasing:
            print("\n  ✅ RUL FEASIBLE: Sequential files suggest temporal ordering!")
    
    # Pattern 2: Specimen/Test ID + Number
    specimen_pattern = df[filename_col].str.extract(r'(specimen|test|sample)[\s_-]?(\d+)', 
                                                     flags=re.IGNORECASE)
    if specimen_pattern[0].notna().sum() > 0:
        specimen_ids = pd.to_numeric(specimen_pattern[1], errors='coerce')
        print(f"\n✓ Specimen/Test Pattern Found:")
        print(f"  Unique specimens: {specimen_ids.nunique()}")
        print(f"  Samples per specimen: {len(df) / specimen_ids.nunique():.1f} avg")
        
        if specimen_ids.nunique() > 5:
            print("\n  ✅ RUL FEASIBLE: Multiple specimens for training!")
    
    # Pattern 3: Load percentage
    load_pattern = df[filename_col].str.extract(r'(\d+)[\s_-]?%', expand=False)
    if load_pattern.notna().sum() > 0:
        load_pcts = pd.to_numeric(load_pattern, errors='coerce')
        print(f"\n✓ Load Percentage Pattern Found:")
        print(f"  Range: [{load_pcts.min():.0f}%, {load_pcts.max():.0f}%]")
        print(f"  Unique levels: {load_pcts.nunique()}")
        print("\n  ✅ RUL FEASIBLE: Can use (100 - load%) as RUL!")
    
    # Pattern 4: Cycle numbers
    cycle_pattern = df[filename_col].str.extract(r'cycle[\s_-]?(\d+)', 
                                                  flags=re.IGNORECASE, expand=False)
    if cycle_pattern.notna().sum() > 0:
        cycles = pd.to_numeric(cycle_pattern, errors='coerce')
        print(f"\n✓ Cycle Number Pattern Found:")
        print(f"  Range: [{cycles.min():.0f}, {cycles.max():.0f}]")
        print(f"  Unique: {cycles.nunique()}")
        print("\n  ✅ RUL FEASIBLE: Can calculate RUL from cycle numbers!")

# ============================================================================
# 2. TEMPORAL TREND ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: TEMPORAL TREND ANALYSIS")
print("="*80)

# Select key damage indicators
damage_indicators = [
    'ch1_energy', 'ch2_energy', 'ch3_energy', 'ch4_energy',
    'ch1_peak_abs', 'ch2_peak_abs', 'ch3_peak_abs', 'ch4_peak_abs',
    'ch1_line_length', 'ch2_line_length', 'ch3_line_length', 'ch4_line_length',
    'ch1_rms', 'ch2_rms', 'ch3_rms', 'ch4_rms'
]

# Check which exist
available_indicators = [col for col in damage_indicators if col in df.columns]

if available_indicators:
    print(f"\nAnalyzing {len(available_indicators)} damage indicators...")
    
    # Assume row order = time order
    df['row_index'] = range(len(df))
    
    # Check for trends
    trending_features = []
    
    for col in available_indicators:
        # Calculate Spearman correlation with index (monotonic trend)
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(df['row_index'], df[col])
        
        if abs(corr) > 0.3 and p_value < 0.05:
            trending_features.append({
                'Feature': col,
                'Correlation': corr,
                'P_Value': p_value,
                'Trend': 'Increasing' if corr > 0 else 'Decreasing'
            })
    
    if trending_features:
        print(f"\n✅ FOUND {len(trending_features)} FEATURES WITH TEMPORAL TRENDS!")
        print("\nTop trending features:")
        trend_df = pd.DataFrame(trending_features).sort_values('Correlation', 
                                                                key=abs, ascending=False)
        print(trend_df.head(10).to_string(index=False))
        
        # Visualize top trends
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, row in trend_df.head(9).iterrows():
            ax = axes[idx]
            col = row['Feature']
            ax.plot(df['row_index'], df[col], linewidth=1, alpha=0.7)
            ax.set_xlabel('Sample Index (Time Proxy)', fontsize=9)
            ax.set_ylabel(col, fontsize=9)
            ax.set_title(f"{col}\nCorr={row['Correlation']:.3f} ({row['Trend']})", 
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_trends.png'), 
                   dpi=300, bbox_inches='tight')
        print("\n✓ Saved: temporal_trends.png")
        plt.close()
        
        print("\n  ✅ RUL FEASIBLE: Clear temporal trends indicate progressive damage!")
    else:
        print("\n  ⚠️ WARNING: No clear temporal trends found.")
        print("     Data may not be time-ordered OR damage is not progressive.")

# ============================================================================
# 3. CREATE EXAMPLE RUL TARGETS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: RUL TARGET CREATION STRATEGIES")
print("="*80)

rul_strategies = []

# Strategy 1: Linear RUL (assuming sequential)
df['RUL_linear'] = len(df) - df['row_index']
rul_strategies.append(('Linear (Sequential)', 'RUL_linear'))
print("\n✓ Strategy 1: Linear RUL (RUL = N - current_index)")
print(f"  Range: [1, {df['RUL_linear'].max()}]")

# Strategy 2: Normalized RUL (0-1)
df['RUL_normalized'] = 1 - (df['row_index'] / len(df))
rul_strategies.append(('Normalized (0-1)', 'RUL_normalized'))
print("\n✓ Strategy 2: Normalized RUL (0=failure, 1=healthy)")
print(f"  Range: [{df['RUL_normalized'].min():.3f}, {df['RUL_normalized'].max():.3f}]")

# Strategy 3: Percentage RUL
df['RUL_percentage'] = 100 * (1 - df['row_index'] / len(df))
rul_strategies.append(('Percentage', 'RUL_percentage'))
print("\n✓ Strategy 3: Percentage RUL")
print(f"  Range: [{df['RUL_percentage'].min():.1f}%, {df['RUL_percentage'].max():.1f}%]")

# Strategy 4: From filename if load pattern exists
if filename_col:
    load_pattern = df[filename_col].str.extract(r'(\d+)%', expand=False)
    if load_pattern.notna().sum() > len(df) * 0.5:  # If >50% have pattern
        df['RUL_from_load'] = 100 - pd.to_numeric(load_pattern, errors='coerce')
        rul_strategies.append(('From Load %', 'RUL_from_load'))
        print("\n✓ Strategy 4: RUL from load percentage in filename")
        print(f"  Range: [{df['RUL_from_load'].min():.1f}%, {df['RUL_from_load'].max():.1f}%]")

# Visualize RUL strategies
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, col) in enumerate(rul_strategies[:4]):
    ax = axes[idx]
    ax.plot(df['row_index'], df[col], linewidth=2)
    ax.set_xlabel('Sample Index', fontweight='bold')
    ax.set_ylabel('RUL', fontweight='bold')
    ax.set_title(f'RUL Strategy: {name}', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add stages
    n = len(df)
    ax.axvspan(0, n*0.3, alpha=0.2, color='green', label='Healthy')
    ax.axvspan(n*0.3, n*0.7, alpha=0.2, color='yellow', label='Degrading')
    ax.axvspan(n*0.7, n, alpha=0.2, color='red', label='Critical')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rul_strategies.png'), dpi=300, bbox_inches='tight')
print("\n✓ Saved: rul_strategies.png")
plt.close()

# ============================================================================
# 4. FEASIBILITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 4: RUL PREDICTION FEASIBILITY ASSESSMENT")
print("="*80)

feasibility_score = 0
max_score = 5
reasons = []

# Check 1: Temporal ordering
if filename_col and sequential.notna().sum() > 0:
    feasibility_score += 1
    reasons.append("✓ Sequential filenames suggest temporal ordering")
else:
    reasons.append("⚠ No clear temporal ordering in filenames")

# Check 2: Damage trends
if len(trending_features) > 5:
    feasibility_score += 2
    reasons.append(f"✓ {len(trending_features)} features show temporal trends")
else:
    reasons.append("⚠ Few features show temporal trends")

# Check 3: Sufficient samples
if len(df) >= 100:
    feasibility_score += 1
    reasons.append(f"✓ {len(df)} samples is sufficient for RUL modeling")
elif len(df) >= 50:
    feasibility_score += 0.5
    reasons.append(f"⚠ {len(df)} samples is marginal for RUL modeling")
else:
    reasons.append(f"✗ {len(df)} samples is too few for RUL modeling")

# Check 4: Feature diversity
if len(available_indicators) >= 8:
    feasibility_score += 1
    reasons.append(f"✓ {len(available_indicators)} damage indicators available")
else:
    reasons.append(f"⚠ Only {len(available_indicators)} damage indicators found")

print(f"\nFEASIBILITY SCORE: {feasibility_score}/{max_score}")
print("\nDetails:")
for reason in reasons:
    print(f"  {reason}")

# Final recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if feasibility_score >= 4:
    recommendation = """
✅ YOUR DATA IS HIGHLY SUITABLE FOR RUL PREDICTION!

RECOMMENDED APPROACH:
1. Use Strategy 1 (Linear RUL) or Strategy 2 (Normalized RUL) as target
2. Select top trending features as inputs
3. Try these models:
   - LSTM (for temporal patterns)
   - Gradient Boosting (for feature interactions)
   - Random Forest (baseline)
4. Validate with time-series cross-validation

NEXT STEPS:
- Run the RUL prediction script (I'll create this)
- Compare model performance
- Tune hyperparameters
- Deploy best model
"""
elif feasibility_score >= 2.5:
    recommendation = """
⚠️ YOUR DATA MAY BE SUITABLE FOR RUL PREDICTION WITH MODIFICATIONS

CONCERNS:
- Limited temporal trends or unclear ordering
- May need feature engineering

RECOMMENDED APPROACH:
1. Verify file ordering represents time progression
2. Create derivative features (differences, rates of change)
3. Start with simpler models (Random Forest, Gradient Boosting)
4. Use careful cross-validation

NEXT STEPS:
- Confirm temporal ordering with domain expert
- Create time-based features
- Test on subset first
"""
else:
    recommendation = """
⚠️ YOUR DATA MAY NOT BE SUITABLE FOR TRADITIONAL RUL PREDICTION

REASONS:
- No clear temporal ordering
- Insufficient temporal trends
- Data may represent different test conditions, not progression

ALTERNATIVE APPROACHES:
1. Classification: Healthy vs. Damaged stages
2. Load capacity prediction (if load info available)
3. Damage severity estimation
4. Anomaly detection

NEXT STEPS:
- Clarify test protocol and data collection
- Consider alternative targets
- Consult with domain expert
"""

print(recommendation)

# Save report
report = f"""
RUL PREDICTION FEASIBILITY REPORT
{'='*80}

DATASET: {FILE_PATH}
Samples: {len(df)}
Features: {df.shape[1]}

FEASIBILITY SCORE: {feasibility_score}/{max_score}

{chr(10).join(reasons)}

{recommendation}

GENERATED FILES:
- temporal_trends.png: Shows features with temporal patterns
- rul_strategies.png: Different RUL target creation methods
- feasibility_report.txt: This report

{'='*80}
"""

with open(os.path.join(OUTPUT_DIR, 'feasibility_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✓ Full report saved to: {OUTPUT_DIR}/feasibility_report.txt")
print(f"\nAll analysis saved to: {OUTPUT_DIR}")
print("="*80)