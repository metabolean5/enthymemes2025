import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter

def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', quotechar='"')
    
    print(f"Initial Dataset Size: {len(df)}")
    
    # Replace missing or empty 'Tree_type' values with 'none'
    df['Tree_type'] = df['Tree_type'].fillna('none')
    df['Tree_type'] = df['Tree_type'].replace('', 'none')
    
    # Merge 'premise' and 'conclusion' into 'implicit'
    df['gold'] = df['gold'].replace({'premise': 'implicit', 'conclusion': 'implicit'})
    
    # Ensure 'gold' column contains only 'implicit' or 'none'
    df = df[df['gold'].isin(['implicit', 'none'])]
    
    # Keep relevant columns
    df = df[['id', 'tweet_text', 'gold']]
    
    # Ensure 'id' is string
    df['id'] = df['id'].astype(str)
    
    print(f"Final Dataset Size after Preprocessing: {len(df)}")
    
    return df

class TextDataset(Dataset):
    def __init__(self, texts, labels, ids, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        id_ = self.ids[idx]

        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'raw_text': text,
            'id': id_
        }

class RobertaClassifier(nn.Module):
    def __init__(self, roberta_model, num_labels=2, dropout=0.1):
        super(RobertaClassifier, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        logits = self.classifier(self.dropout(cls_output))
        return logits

def train_model(model, train_loader, val_loader, device, fold, epochs=4, lr=2e-5, patience=4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        true_labels = []
        preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                preds.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Fold {fold}, Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)

    # Final validation metrics
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    true_labels = []
    preds = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch['id']

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
            val_ids.extend(ids)

    val_accuracy = 100 * correct / total
    return val_loss / len(val_loader), val_accuracy, true_labels, preds, val_ids

def cross_validate_model(df, n_splits=5):
    label_encoder = LabelEncoder()
    df['gold'] = label_encoder.fit_transform(df['gold'])
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    texts = df['tweet_text'].values
    labels = df['gold'].values
    ids = df['id'].values

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Define 20 random states
    random_states = list(range(200, 220))
    all_results = []
    id_predictions = {id_: [] for id_ in ids}

    for rs in random_states:
        print(f"\n=== Random State {rs} ===")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
        fold = 1
        val_losses = []
        val_accuracies = []
        all_true_labels = []
        all_preds = []

        for train_idx, val_idx in kf.split(texts):
            print(f"\n=== Random State {rs}, Fold {fold}/{n_splits} ===")
            
            X_train, X_val = texts[train_idx], texts[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            train_ids, val_ids = ids[train_idx], ids[val_idx]

            # Create datasets
            train_dataset = TextDataset(X_train, y_train, train_ids, tokenizer)
            val_dataset = TextDataset(X_val, y_val, val_ids, tokenizer)

            # Print sample inputs (up to 3 samples from training set, only for first random state)
            if rs == random_states[0]:
                print("\nSample Inputs (Training Set):")
                for i in range(min(3, len(train_dataset))):
                    sample = train_dataset[i]
                    print(f"\nSample {i+1}:")
                    print(f"Raw Text: {sample['raw_text']}")
                    print(f"Label: {sample['labels'].item()} ({label_encoder.inverse_transform([sample['labels'].item()])[0]})")
                    print(f"Input IDs (first 10): {sample['input_ids'][:10].numpy()}")
                    print(f"Attention Mask (first 10): {sample['attention_mask'][:10].numpy()}")

            # Initialize model
            roberta_model = RobertaModel.from_pretrained('roberta-base')
            model = RobertaClassifier(roberta_model, num_labels=2)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)

            # Train and evaluate
            val_loss, val_accuracy, true_labels, preds, val_ids = train_model(model, train_loader, val_loader, device, fold)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            all_true_labels.extend(true_labels)
            all_preds.extend(preds)

            # Store predictions for each ID
            for id_, pred in zip(val_ids, preds):
                id_predictions[id_].append(label_encoder.inverse_transform([pred])[0])

            # Print fold-specific classification report
            print(f"\nFold {fold} Classification Report (Random State {rs}):")
            print(classification_report(true_labels, preds, target_names=['implicit', 'none']))

            print(f"\nFold {fold} Results (Random State {rs}):")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.2f}%")
            fold += 1

        # Overall classification report for this random state
        print(f"\n=== Random State {rs} Classification Report (Across All Folds) ===")
        clf_report = classification_report(all_true_labels, all_preds, target_names=['implicit', 'none'], output_dict=True)
        print(classification_report(all_true_labels, all_preds, target_names=['implicit', 'none']))

        # Store results for this random state
        all_results.append({
            'random_state': rs,
            'avg_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'avg_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'classification_report': clf_report
        })

        # Report average metrics for this random state
        print(f"\n=== Random State {rs} Cross-Validation Summary ===")
        print(f"Average Validation Loss: {np.mean(val_losses):.4f} (±{np.std(val_losses):.4f})")
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
    results_df.to_csv('prediction_results_roberta_text_only.csv', index=False)
    print("\nSaved prediction results to 'prediction_results_roberta_text_only.csv'")

    # Summary of all random states
    print("\n=== Summary Across All Random States ===")
    print(f"{'Random State':<15} {'Avg Val Loss':<15} {'Std Val Loss':<15} {'Avg Val Acc':<15} {'Std Val Acc':<15} "
          f"{'Avg Precision':<15} {'Avg Recall':<15} {'Avg F1':<15}")
    print("-" * 110)
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
        print(f"{result['random_state']:<15} {result['avg_val_loss']:<15.4f} {result['std_val_loss']:<15.4f} "
              f"{result['avg_val_accuracy']:<15.2f} {result['std_val_accuracy']:<15.2f} "
              f"{macro_precision:<15.4f} {macro_recall:<15.4f} {macro_f1:<15.4f}")

    # Average metrics across all random states
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
    print(f"\nAverage Validation Loss: {np.mean([r['avg_val_loss'] for r in all_results]):.4f} "
          f"(±{np.std([r['avg_val_loss'] for r in all_results]):.4f})")
    print(f"Average Validation Accuracy: {np.mean([r['avg_val_accuracy'] for r in all_results]):.2f}% "
          f"(±{np.std([r['avg_val_accuracy'] for r in all_results]):.2f})")

    # Detailed classification report for each random state
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
    # Load data
    file_path = 'tropes_CLEANrst_immigration_mpno_treetypes_1.csv'
    df = load_data(file_path)

    print(f"Total Dataset Size: {len(df)}")
    print(f"Class Distribution: {df['gold'].value_counts().to_dict()}")

    # Perform 5-fold cross-validation over 20 random states
    cross_validate_model(df)

if __name__ == '__main__':
    main()