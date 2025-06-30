import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import sparse
import pickle
from collections import Counter

def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', quotechar='"')
    
    print(f"Initial Dataset Size: {len(df)}")
    
    # Replace missing or empty 'Tree_type' values with 'none'
    df['Tree_type'] = df['Tree_type'].fillna('none')
    df['Tree_type'] = df['Tree_type'].replace('', 'none')
    
    # Merge 'conclusion' and 'premise' into 'implicit'
    df['gold'] = df['gold'].replace({'conclusion': 'implicit', 'premise': 'implicit'})
    
    # Drop rows with missing or invalid 'gold'
    df = df[df['gold'].isin(['implicit', 'none'])]
    
    # Load rst_features (expecting 11-dimensional vectors)
    with open('rst_features4.pkl', 'rb') as file:
        rst_features = pickle.load(file)
    
    # Adjust keys to match id format
    #rst_features = {f"{k}.0": v for k, v in rst_features1.items()}
    
    # Keep relevant columns
    df = df[['id', 'tweet_text', 'gold']]
    
    # Ensure 'id' is string to match rst_features keys
    df['id'] = df['id'].astype(str)
    
    # Verify that all IDs in df have corresponding rst_features
    missing_ids = [id_ for id_ in df['id'] if id_ not in rst_features]
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in rst_features: {missing_ids[:5]}...")
        df = df[df['id'].isin(rst_features.keys())]
    
    print(f"Final Dataset Size after Preprocessing: {len(df)}")
    
    return df, rst_features

def train_model(X_train_text, X_train_rst, y_train, X_val_text, X_val_rst, y_val, val_ids, fold):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.95, min_df=2)
    
    # Fit and transform training text data
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    
    # Convert RST features to numpy arrays (stack list of vectors)
    X_train_rst = np.vstack([X_train_rst[i] for i in range(len(X_train_rst))])
    X_val_rst = np.vstack([X_val_rst[i] for i in range(len(X_val_rst))])
    
    # Combine TF-IDF and RST features
    X_train_combined = sparse.hstack([X_train_tfidf, X_train_rst])
    X_val_combined = sparse.hstack([X_val_tfidf, X_val_rst])
    
    # Initialize and train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_combined, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val_combined)
    
    # Calculate accuracy
    accuracy = 100 * (y_pred == y_val).mean()
    
    return accuracy, y_val, y_pred, val_ids

def cross_validate_model(df, rst_features, n_splits=5):
    label_encoder = LabelEncoder()
    df['gold'] = label_encoder.fit_transform(df['gold'])
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    texts = df['tweet_text'].values
    ids = df['id'].values
    labels = df['gold'].values
    
    # Map RST features to dataset rows using ids
    rst_features_array = [rst_features[id_] for id_ in ids]

    random_states = list(range(200, 220))
    all_results = []
    # Store predictions for each ID across all random states
    id_predictions = {id_: [] for id_ in ids}

    for rs in random_states:
        print(f"\n=== Random State {rs} ===")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
        fold = 1
        val_accuracies = []
        all_true_labels = []
        all_preds = []

        for train_idx, val_idx in kf.split(texts):
            print(f"\n=== Random State {rs}, Fold {fold}/{n_splits} ===")
            
            X_train_text, X_val_text = texts[train_idx], texts[val_idx]
            X_train_rst, X_val_rst = [rst_features_array[i] for i in train_idx], [rst_features_array[i] for i in val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            val_ids = ids[val_idx]

            # Print sample inputs (up to 3 samples from training set)
            if rs == random_states[0]:
                print("\nSample Inputs (Training Set):")
                for i in range(min(3, len(X_train_text))):
                    print(f"\nSample {i+1}:")
                    print(f"Raw Text: {X_train_text[i]}")
                    print(f"RST Features (11 dims): {X_train_rst[i]}")
                    print(f"Label: {y_train[i]} ({label_encoder.inverse_transform([y_train[i]])[0]})")

            accuracy, true_labels, preds, val_ids = train_model(
                X_train_text, X_train_rst, y_train, X_val_text, X_val_rst, y_val, val_ids, fold
            )
            val_accuracies.append(accuracy)
            all_true_labels.extend(true_labels)
            all_preds.extend(preds)

            # Store predictions for each ID in this fold
            for id_, pred in zip(val_ids, preds):
                id_predictions[id_].append(label_encoder.inverse_transform([pred])[0])

            print(f"\nFold {fold} Classification Report (Random State {rs}):")
            print(classification_report(true_labels, preds, target_names=['implicit', 'none']))

            print(f"\nFold {fold} Results (Random State {rs}):")
            print(f"Val Accuracy: {accuracy:.2f}%")
            fold += 1

        print(f"\n=== Random State {rs} Classification Report (Across All Folds) ===")
        clf_report = classification_report(all_true_labels, all_preds, target_names=['implicit', 'none'], output_dict=True)
        print(classification_report(all_true_labels, all_preds, target_names=['implicit', 'none']))

        all_results.append({
            'random_state': rs,
            'avg_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'classification_report': clf_report
        })

        print(f"\n=== Random State {rs} Cross-Validation Summary ===")
        print(f"Average Validation Accuracy: {np.mean(val_accuracies):.2f}% (±{np.std(val_accuracies):.2f})")
        print(f"Label Mapping: {label_mapping}")

    # Create DataFrame for most predicted labels
    results_df = pd.DataFrame({
        'id': ids,
        'most_predicted_label': [
            Counter(id_predictions[id_]).most_common(1)[0][0] if id_predictions[id_] else 'none'
            for id_ in ids
        ]
    })
    results_df.to_csv('prediction_results_rst.csv', index=False)
    print("\nSaved prediction results to 'prediction_results.csv'")

    print("\n=== Summary Across All Random States ===")
    print(f"{'Random State':<15} {'Avg Val Acc':<15} {'Std Val Acc':<15} "
          f"{'Avg Precision':<15} {'Avg Recall':<15} {'Avg F1':<15}")
    print("-" * 90)
    precisions = []
    recalls = []
    f1_scores = []
    implicit_precisions = []
    implicit_recalls = []
    implicit_f1_scores = []
    none_precisions = []
    none_recalls = []
    none_f1_scores = []
    for result in all_results:
        macro_precision = result['classification_report']['macro avg']['precision']
        macro_recall = result['classification_report']['macro avg']['recall']
        macro_f1 = result['classification_report']['macro avg']['f1-score']
        implicit_precision = result['classification_report']['implicit']['precision']
        implicit_recall = result['classification_report']['implicit']['recall']
        implicit_f1 = result['classification_report']['implicit']['f1-score']
        none_precision = result['classification_report']['none']['precision']
        none_recall = result['classification_report']['none']['recall']
        none_f1 = result['classification_report']['none']['f1-score']
        precisions.append(macro_precision)
        recalls.append(macro_recall)
        f1_scores.append(macro_f1)
        implicit_precisions.append(implicit_precision)
        implicit_recalls.append(implicit_recall)
        implicit_f1_scores.append(implicit_f1)
        none_precisions.append(none_precision)
        none_recalls.append(none_recall)
        none_f1_scores.append(none_f1)
        print(f"{result['random_state']:<15} {result['avg_val_accuracy']:<15.2f} {result['std_val_accuracy']:<15.2f} "
              f"{macro_precision:<15.4f} {macro_recall:<15.4f} {macro_f1:<15.4f}")

    print("\n=== Average Metrics Across All Random States ===")
    print(f"Average Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
    print(f"Average Recall: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
    print(f"Average F1-Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
    print(f"\nAverage Metrics for 'implicit' Label:")
    print(f"  Precision: {np.mean(implicit_precisions):.4f} (±{np.std(implicit_precisions):.4f})")
    print(f"  Recall: {np.mean(implicit_recalls):.4f} (±{np.std(implicit_recalls):.4f})")
    print(f"  F1-Score: {np.mean(implicit_f1_scores):.4f} (±{np.std(implicit_f1_scores):.4f})")
    print(f"\nAverage Metrics for 'none' Label:")
    print(f"  Precision: {np.mean(none_precisions):.4f} (±{np.std(none_precisions):.4f})")
    print(f"  Recall: {np.mean(none_recalls):.4f} (±{np.std(none_recalls):.4f})")
    print(f"  F1-Score: {np.mean(none_f1_scores):.4f} (±{np.std(none_f1_scores):.4f})")
    print(f"\nAverage Validation Accuracy: {np.mean([r['avg_val_accuracy'] for r in all_results]):.2f}% "
          f"(±{np.std([r['avg_val_accuracy'] for r in all_results]):.2f})")

    print("\n=== Detailed Classification Reports ===")
    for result in all_results:
        rs = result['random_state']
        print(f"\nRandom State {rs}:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 55)
        for cls in ['implicit', 'none']:
            metrics = result['classification_report'][cls]
            print(f"{cls:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
        print(f"{'Accuracy':<15} {'':<10} {'':<10} {result['classification_report']['accuracy']:<10.4f} "
              f"{result['classification_report']['macro avg']['support']:<10}")

def main():
    file_path = 'tropes_CLEANrst_immigration_mpno_treetypes_1.csv'
    df, rst_features = load_data(file_path)

    print(f"Total Dataset Size: {len(df)}")
    print(f"Class Distribution: {df['gold'].value_counts().to_dict()}")
    print(f"RST Feature Vector Length: {len(rst_features[next(iter(rst_features))])} (11 dimensions: 5 depth sums, 6 structural features)")

    cross_validate_model(df, rst_features)

if __name__ == '__main__':
    main()