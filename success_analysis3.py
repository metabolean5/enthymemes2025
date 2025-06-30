import pandas as pd
import numpy as np
import pickle

def load_data(csv_path, rst_pickle_path):
    df = pd.read_csv(csv_path, sep=',', quotechar='"')
    df['id'] = df['id'].astype(str)
    print(f"CSV Loaded: {len(df)} rows")
    
    df['gold'] = df['gold'].replace({'conclusion': 'implicit', 'premise': 'implicit'})
    df = df[df['gold'].isin(['implicit', 'none'])]
    print(f"Dataset Size after Gold Label Filtering: {len(df)}")
    
    disagreements = (df['rst'] != df['tropes']).sum()
    print(f"Number of Cases where RST and Tropes Disagree: {disagreements}")
    
    df['rst_correct'] = df['rst'] == df['gold']
    df['tropes_incorrect'] = df['tropes'] != df['gold']
    df = df[df['rst_correct'] & df['tropes_incorrect']]
    print(f"Cases where RST is Correct and Tropes are Incorrect: {len(df)}")
    
    none_ids = df[df['gold'] == 'none'][['id', 'clean_rst_tree']]
    implicit_ids = df[df['gold'] == 'implicit'][['id', 'clean_rst_tree']]
    none_ids.to_csv('none_instance_ids.csv', index=False)
    implicit_ids.to_csv('implicit_instance_ids.csv', index=False)
    print(f"Saved {len(none_ids)} none instance IDs with RST trees to 'none_instance_ids.csv'")
    print(f"Saved {len(implicit_ids)} implicit instance IDs with RST trees to 'implicit_instance_ids.csv'")
    
    with open(rst_pickle_path, 'rb') as file:
        rst_features = pickle.load(file)
    
    missing_ids = [id_ for id_ in df['id'] if id_ not in rst_features]
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in rst_features: {missing_ids[:5]}...")
        df = df[df['id'].isin(rst_features.keys())]
    
    rst_features_array = np.array([rst_features[id_] for id_ in df['id']])
    print(f"Final Dataset Size: {len(df)}")
    print(f"RST Feature Vector Length: {rst_features_array.shape[1]}")
    
    return df, rst_features_array

def get_rst_feature_names():
    relation_types = [
        'attribution', 'background', 'cause', 'comparison', 'condition', 'contrast',
        'elaboration', 'enablement', 'evaluation', 'explanation', 'joint', 'manner-means',
        'same-unit', 'satellite', 'temporal', 'textualorganization', 'topic-comment'
    ]
    feature_names = [f"presence_{rel}" for rel in relation_types]
    feature_names.append('depth_sum_unknown')
    feature_names.extend([
        'num_segments', 'num_relations', 'avg_segment_length', 'max_depth',
        'nucleus_spans', 'satellite_spans', 'nucleus_satellite_relations',
        'nucleus_nucleus_relations', 'satellite_satellite_relations',
        'nucleus_to_satellite_ratio', 'discourse_marker_presence'
    ])
    for i in range(len(feature_names), 31):
        feature_names.append(f"unknown_feature_{i - len(feature_names) + 1}")
    return feature_names

def discretize_rst_features(rst_features):
    discretized_features = np.zeros_like(rst_features, dtype=float)
    for idx in range(17):
        discretized_features[:, idx] = (rst_features[:, idx] > 0).astype(float)
    for idx in range(17, rst_features.shape[1]):
        discretized_features[:, idx] = np.round(rst_features[:, idx]).astype(float)
    return discretized_features

def print_feature_table(df, rst_features, feature_names):
    none_mask = df['gold'] == 'none'
    implicit_mask = df['gold'] == 'implicit'
    
    none_features = rst_features[none_mask]
    implicit_features = rst_features[implicit_mask]
    
    none_mean = np.mean(none_features, axis=0) if len(none_features) > 0 else np.zeros(len(feature_names))
    implicit_mean = np.mean(implicit_features, axis=0) if len(implicit_features) > 0 else np.zeros(len(feature_names))
    
    heatmap_data = pd.DataFrame({
        'None': none_mean,
        'Implicit': implicit_mean
    }, index=feature_names)
    
    # Sort by difference in means (None - Implicit)
    mean_diff = heatmap_data['None'] - heatmap_data['Implicit']
    sorted_indices = mean_diff.sort_values(ascending=False).index
    heatmap_data = heatmap_data.loc[sorted_indices]
    
    # Print table with error handling
    print("Feature                          | None Mean | Implicit Mean")
    print("---------------------------------|-----------|--------------")
    skipped_features = []
    for feature in heatmap_data.index:
        try:
            none_val = float(heatmap_data.loc[feature, 'None'])
            implicit_val = float(heatmap_data.loc[feature, 'Implicit'])
            if pd.isna(none_val) or pd.isna(implicit_val):
                skipped_features.append(feature)
                continue
            print(f"{feature.ljust(32)}| {f'{none_val:.2f}'.center(10)}| {f'{implicit_val:.2f}'.center(13)}")
        except (TypeError, ValueError) as e:
            skipped_features.append(feature)
            print(f"Warning: Skipped feature '{feature}' due to error: {str(e)}")
    
    if skipped_features:
        print(f"\nSkipped {len(skipped_features)} features due to errors: {', '.join(skipped_features)}")

def main():
    csv_path = 'tropes-rst_error_analysis.csv'
    rst_pickle_path = 'rst_features4.pkl'
    
    df, rst_features = load_data(csv_path, rst_pickle_path)
    
    print('Generating RST feature names...')
    rst_feature_names = get_rst_feature_names()
    print(f"Total RST Feature Count: {len(rst_feature_names)}")
    
    rst_features_discretized = discretize_rst_features(rst_features)
    
    print("\nGenerating feature table...")
    print_feature_table(df, rst_features_discretized, rst_feature_names)

if __name__ == '__main__':
    main()