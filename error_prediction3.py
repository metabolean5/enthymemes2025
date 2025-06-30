import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
import shap

def load_data(csv_path, prediction_csv_path, rst_pickle_path):
    # Load main dataset with gold labels and tropes
    df = pd.read_csv(csv_path, sep=',', quotechar='"')
    
    print(f"Initial Dataset Size: {len(df)}")
    
    # Replace missing or empty 'Tree_type' values with 'none'
    df['Tree_type'] = df['Tree_type'].fillna('none')
    df['Tree_type'] = df['Tree_type'].replace('', 'none')
    
    # Merge 'conclusion' and 'premise' into 'implicit'
    df['gold'] = df['gold'].replace({'conclusion': 'implicit', 'premise': 'implicit'})
    
    # Drop rows with missing or invalid 'gold'
    df = df[df['gold'].isin(['implicit', 'none'])]
    
    # Define specified tropes
    specified_tropes = [
        'time_will_tell', 'distrust_experts', 'too_fast_early_dev', 
        'natural_traditional_is_better', 'liberty_freedom', 'hidden_motives', 
        'scapegoat', 'defend_the_weak', 'wicked_fairness', 'none'
    ]
    
    # Identify trope columns
    trope_columns = [col for col in df.columns if col in specified_tropes]
    
    # Keep relevant columns
    df = df[['id', 'tweet_text', 'gold'] + trope_columns]
    
    # Ensure 'id' is string
    df['id'] = df['id'].astype(str)
    
    # Load predicted labels
    pred_df = pd.read_csv(prediction_csv_path, sep=',', quotechar='"')
    pred_df['id'] = pred_df['id'].astype(str)
    
    # Merge with main dataframe to get gold and predicted labels
    df = df.merge(pred_df[['id', 'most_predicted_label']], on='id', how='inner')
    print(f"Dataset Size after Merging with Predicted Labels: {len(df)}")
    
    # Compute success/error labels (success if prediction is correct)
    df['success_error'] = df.apply(
        lambda row: 'success' if row['gold'] == row['most_predicted_label'] else 'error', axis=1
    )
    print(f"Success/Error Distribution: {df['success_error'].value_counts().to_dict()}")
    
    # Load RST features
    with open(rst_pickle_path, 'rb') as file:
        rst_features = pickle.load(file)
    
    # Verify that all IDs in df have corresponding rst_features
    missing_ids = [id_ for id_ in df['id'] if id_ not in rst_features]
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in rst_features: {missing_ids[:5]}...")
        df = df[df['id'].isin(rst_features.keys())]
    
    # Extract trope features as numerical vectors
    trope_features = df[trope_columns].values
    
    # Map RST features to dataset rows using IDs
    rst_features_array = np.array([rst_features[id_] for id_ in df['id']])
    
    print(f"Final Dataset Size after Preprocessing: {len(df)}")
    print(f"RST Feature Vector Length: {rst_features_array.shape[1]}")
    print(f"Number of Trope Features: {len(trope_columns)}")
    
    return df, rst_features_array, trope_features, trope_columns

def main():
    # File paths
    csv_path = 'tropes_CLEANrst_immigration_mpno_treetypes_1.csv'
    prediction_csv_path = 'prediction_results_roberta_text_only.csv'
    rst_pickle_path = 'rst_features4.pkl'
    model_path = 'XGBClassifier_rst_tropes_success_error.joblib'
    
    # Define relation-to-index mapping
    relation_to_index = {
        'attribution': 0, 'background': 1, 'cause': 2, 'comparison': 3, 'condition': 4,
        'contrast': 5, 'elaboration': 6, 'enablement': 7, 'evaluation': 8, 'explanation': 9,
        'joint': 10, 'manner-means': 11, 'same-unit': 12, 'satellite': 13, 'temporal': 14,
        'textualorganization': 15, 'topic-comment': 16
    }
    
    # Load data
    df, rst_features, trope_features, trope_columns = load_data(csv_path, prediction_csv_path, rst_pickle_path)
    
    # Define RST feature names (including leaf_nodes)
    rst_feature_names = (
        [f"depth_sum_{rel}" for rel in relation_to_index.keys()] + 
        [
            'depth_sum_unknown',
            'num_segments',
            'num_relations',
            'avg_segment_length',
            'max_depth',
            'nucleus_spans',
            'satellite_spans',
            'nucleus_satellite_relations',
            'nucleus_nucleus_relations',
            'satellite_satellite_relations',
            'leaf_nodes',
            'internal_nodes',
            'nucleus_to_satellite_ratio',
            'discourse_marker_presence'
        ]
    )
    print(f"RST Feature Names: {rst_feature_names}")
    print(f"Number of RST Features: {len(rst_feature_names)}")
    print(f"RST Features Shape: {rst_features.shape}")
    
    # Verify RST feature count
    if len(rst_feature_names) != rst_features.shape[1]:
        print(f"Warning: Number of RST feature names ({len(rst_feature_names)}) does not match "
              f"rst_features shape ({rst_features.shape[1]}). Adding placeholder for extra feature.")
        while len(rst_feature_names) < rst_features.shape[1]:
            rst_feature_names.append(f"unknown_feature_{len(rst_feature_names)}")
        print(f"Updated RST Feature Names: {rst_feature_names}")
    
    # Combine RST and trope features
    X = np.hstack([rst_features, trope_features])
    
    # Define feature names for visualization
    feature_names = rst_feature_names + trope_columns
    print(f"Trope Feature Names: {trope_columns}")
    print(f"Total Number of Features: {len(feature_names)}")
    print(f"X Shape: {X.shape}")
    
    # Verify feature alignment
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"Feature names length ({len(feature_names)}) does not match X shape ({X.shape[1]})")
    
    # Prepare labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['success_error'])
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(f"Label Mapping: {label_mapping}")
    
    # Set up 5-fold cross-validation
    print('Setting up 5-fold cross-validation...')
    seed = 10
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Lists to store SHAP values and model performance
    shap_values_success_folds = []
    shap_values_error_folds = []
    accuracies = []
    
    # Perform cross-validation
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f'\nProcessing Fold {fold}/{n_folds}...')
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train XGBoost model
        print('Training model...')
        model = XGBClassifier(random_state=seed)
        model.fit(X_train, y_train)
        
        # Evaluate model performance
        print('Evaluating model performance...')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Compute SHAP values
        print('Computing SHAP values...')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        print(f"SHAP values shape for fold {fold}: {np.shape(shap_values)}")
        
        # Verify SHAP values shape
        if shap_values.shape[1] != X.shape[1]:
            raise ValueError(f"SHAP values shape ({shap_values.shape[1]}) does not match X shape ({X.shape[1]})")
        
        # Store SHAP values for success and error
        shap_values_success_folds.append(shap_values)  # For success (positive class)
        shap_values_error_folds.append(-shap_values)  # For error (negative class)
        
        fold += 1
    
    # Save the last model
    joblib.dump(model, model_path)
    print(f"Last model saved to {model_path}")
    
    # Report average accuracy
    print(f"\nAverage Accuracy across {n_folds} folds: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    # Aggregate SHAP values across folds
    print('Aggregating SHAP values across folds...')
    
    # Compute mean absolute SHAP values for each fold
    success_importances_folds = [np.abs(fold_shap).mean(axis=0) for fold_shap in shap_values_success_folds]
    error_importances_folds = [np.abs(fold_shap).mean(axis=0) for fold_shap in shap_values_error_folds]
    
    # Average across folds
    success_importances = np.mean(success_importances_folds, axis=0)
    error_importances = np.mean(error_importances_folds, axis=0)
    
    # Compute mean SHAP values for directionality (using last fold)
    success_directions = np.mean(shap_values_success_folds[-1], axis=0)
    error_directions = np.mean(shap_values_error_folds[-1], axis=0)
    
    print(f"Aggregated Mean Absolute SHAP Importances for Success: {success_importances.tolist()}")
    print(f"Aggregated Mean Absolute SHAP Importances for Error: {error_importances.tolist()}")
    print(f"Mean SHAP Directions for Success: {success_directions.tolist()}")
    print(f"Mean SHAP Directions for Error: {error_directions.tolist()}")
    
    # Function to plot SHAP feature importance
    def plot_shap_importance(shap_importances, shap_directions, class_name, output_filename, feature_names):
        if class_name.lower() == 'success':
            title_suffix = "(Increasing Probability of Success)"
        elif class_name.lower() == 'error':
            title_suffix = "(Increasing Probability of Error)"
        else:
            raise ValueError(f"Unknown class name: {class_name}")
        
        # Verify input lengths
        if len(shap_importances) != len(feature_names) or len(shap_directions) != len(feature_names):
            raise ValueError(f"Length mismatch: shap_importances ({len(shap_importances)}), "
                             f"shap_directions ({len(shap_directions)}), feature_names ({len(feature_names)})")
        
        # Remove leaf_nodes from plotting
        keep_indices = [i for i, name in enumerate(feature_names) if name != 'leaf_nodes']
        if len(keep_indices) < len(feature_names):
            print(f"Removing 'leaf_nodes' from plot for {class_name}")
            feature_names = [feature_names[i] for i in keep_indices]
            shap_importances = shap_importances[keep_indices]
            shap_directions = shap_directions[keep_indices]
        
        # Sort features by aggregated importance
        sorted_indices = np.argsort(shap_importances)[::-1]  # Descending order
        
        # Ensure indices are valid
        if max(sorted_indices) >= len(feature_names):
            raise ValueError(f"Invalid index in sorted_indices: max index {max(sorted_indices)}, "
                             f"feature_names length {len(feature_names)}")
        
        sorted_labels = [feature_names[i] for i in sorted_indices]
        sorted_importances = shap_importances[sorted_indices]
        sorted_directions = shap_directions[sorted_indices]
        
        # Keep top 10 features
        sorted_labels = sorted_labels[:10]
        sorted_importances = sorted_importances[:10]
        sorted_directions = sorted_directions[:10]
        
        # Define colors based on feature type and direction
        colors = []
        for label, direction in zip(sorted_labels, sorted_directions):
            if label.startswith('depth_sum_') or label in [
                'num_segments', 'num_relations', 'avg_segment_length', 'max_depth',
                'nucleus_spans', 'satellite_spans', 'nucleus_satellite_relations',
                'nucleus_nucleus_relations', 'satellite_satellite_relations',
                'internal_nodes', 'nucleus_to_satellite_ratio',
                'discourse_marker_presence'
            ]:
                # RST features
                base_color = 'skyblue' if direction >= 0 else 'steelblue'
            else:
                # Trope features
                base_color = 'salmon' if direction >= 0 else 'darkred'
            colors.append(base_color)
        
        # Plotting
        print(f'Generating SHAP feature importance plot for {class_name}...')
        fig, ax = plt.subplots(figsize=(5.5, 6))  # More compact for top 10 features
        
        # Create horizontal bar plot with slimmer bars
        bars = ax.barh(range(len(sorted_importances)), sorted_importances, align='center', color=colors, height=0.3)
        
        # Add value labels on bars with direction info
        for i, bar in enumerate(bars):
            width = bar.get_width()
            direction = sorted_directions[i]
            label = f'{sorted_importances[i]:.4f} ({"+" if direction >= 0 else "-"})'
            ax.text(x=width + 0.001, y=bar.get_y() + bar.get_height()/2, s=label,
                    ha='left', va='center', fontsize=7)
        
        # Customize plot
        ax.set_yticks(range(len(sorted_importances)))
        ax.set_yticklabels(sorted_labels, fontsize=9)  # Larger feature labels
        ax.set_xlabel('Mean Absolute SHAP Value', fontsize=8)
        ax.set_ylabel('Features', fontsize=8)
        ax.set_title(f'SHAP Feature Importances for {class_name}\n{title_suffix}', fontsize=9)
        ax.invert_yaxis()  # Highest importance at top
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='RST Features (Positive)'),
            Patch(facecolor='steelblue', label='RST Features (Negative)'),
            Patch(facecolor='salmon', label='Trope Features (Positive)'),
            Patch(facecolor='darkred', label='Trope Features (Negative)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=7)
        
        # Adjust layout for tight fit
        plt.tight_layout(pad=0.8)  # Tighter layout
        
        # Save the plot
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        print(f"Plot saved to '{output_filename}'")
        plt.close()
    
    # Plot aggregated SHAP importances for both classes
    print('Generating aggregated SHAP plots...')
    plot_shap_importance(success_importances, success_directions, 'Success', 'shap_feature_importances_success_5fold.png', feature_names)
    plot_shap_importance(error_importances, error_directions, 'Error', 'shap_feature_importances_error_5fold.png', feature_names)

if __name__ == '__main__':
    main()