# src/evaluate.py

from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

def evaluate_model(y_true, y_pred, threshold=0.3):
    """
    Multi label classification evaluation:
    - y_pred: sigmoid outputs (float)
    - y_true: binary ground truth
    """
    y_pred_bin = (y_pred >= threshold).astype(int)

    print("Accuracy Score:", accuracy_score(y_true, y_pred_bin))
    print("Micro F1 Score:", f1_score(y_true, y_pred_bin, average='micro'))
    print("Macro F1 Score:", f1_score(y_true, y_pred_bin, average='macro'))
    print("Precision Score:", precision_score(y_true, y_pred_bin, average='micro'))
    print("Recall Score:", recall_score(y_true, y_pred_bin, average='micro'))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_bin, zero_division=0))

