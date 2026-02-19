import pandas as pd
import numpy as np

# Read all datasets
print("="*80)
print("THAI-MOD Dataset Analysis for Person 1")
print("="*80)

datasets = {
    'dataset1': 'datasets/dataset1.csv',
    'dataset2': 'datasets/dataset2.csv',
    'dataset3': 'datasets/dataset3.csv',
}

all_stats = []

for name, path in datasets.items():
    print(f"\n{'='*80}")
    print(f"Analyzing {name.upper()}")
    print(f"{'='*80}")
    
    df = pd.read_csv(path)
    
    # Basic info
    print(f"\n📊 Basic Information:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    
    # Get label column (first column)
    label_col = df.columns[0]
    text_col = df.columns[1]
    
    # Label distribution
    print(f"\n📈 Label Distribution:")
    label_counts = df[label_col].value_counts()
    print(label_counts)
    
    # Calculate percentages
    total = len(df)
    print(f"\n📊 Percentages:")
    for label, count in label_counts.items():
        pct = (count / total) * 100
        print(f"  - {label}: {count:,} ({pct:.2f}%)")
    
    # Toxic rate
    toxic_labels = ['tox', 'toxic', 'neg', 'negative']
    toxic_count = sum(label_counts.get(label, 0) for label in toxic_labels)
    toxic_rate = (toxic_count / total) * 100
    
    print(f"\n🔴 Toxic Rate: {toxic_rate:.2f}%")
    
    # Text length statistics
    if 'length' in df.columns:
        print(f"\n📏 Text Length Statistics:")
        print(df['length'].describe())
    else:
        # Calculate length
        df['text_length'] = df[text_col].astype(str).str.len()
        print(f"\n📏 Text Length Statistics (character count):")
        print(df['text_length'].describe())
    
    # Missing values
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")
    
    # Store stats
    all_stats.append({
        'Dataset': name,
        'Total Samples': total,
        'Labels': dict(label_counts),
        'Toxic Rate (%)': toxic_rate,
        'Columns': df.columns.tolist()
    })

# Combined analysis
print(f"\n{'='*80}")
print("COMBINED ANALYSIS")
print(f"{'='*80}")

# Load combined dataset
df1 = pd.read_csv('datasets/dataset1.csv')
df2 = pd.read_csv('datasets/dataset2.csv')
df3 = pd.read_csv('datasets/dataset3.csv')

# Standardize column names
df1.columns = ['category', 'texts', 'length']
df2.columns = ['category', 'texts', 'length']
df3.columns = ['category', 'texts', 'length']

# Add source
df1['source'] = 'dataset1'
df2['source'] = 'dataset2'
df3['source'] = 'dataset3'

# Combine
combined = pd.concat([df1, df2, df3], ignore_index=True)

print(f"\n📊 Combined Dataset:")
print(f"  - Total samples: {len(combined):,}")
print(f"  - Total columns: {combined.columns.tolist()}")

print(f"\n📈 Combined Label Distribution:")
combined_labels = combined['category'].value_counts()
print(combined_labels)

print(f"\n📊 Combined Percentages:")
for label, count in combined_labels.items():
    pct = (count / len(combined)) * 100
    print(f"  - {label}: {count:,} ({pct:.2f}%)")

# Class imbalance ratio
toxic_labels = ['tox', 'toxic', 'neg', 'negative']
non_toxic_labels = ['non-tox', 'non-toxic', 'neu', 'neutral', 'pos', 'positive']

toxic_total = sum(combined_labels.get(label, 0) for label in toxic_labels)
non_toxic_total = sum(combined_labels.get(label, 0) for label in non_toxic_labels)

print(f"\n⚖️ Class Imbalance:")
print(f"  - Toxic: {toxic_total:,} ({toxic_total/len(combined)*100:.2f}%)")
print(f"  - Non-toxic: {non_toxic_total:,} ({non_toxic_total/len(combined)*100:.2f}%)")
print(f"  - Imbalance Ratio: 1:{non_toxic_total/toxic_total:.2f}")

# Distribution by source
print(f"\n📦 Distribution by Source:")
source_dist = combined['source'].value_counts()
print(source_dist)

print(f"\n✅ Analysis Complete!")
