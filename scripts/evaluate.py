import numpy as np
from sklearn.metrics import f1_score, jaccard_score

def evaluate_model(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = (y_pred.flatten() > 0.5).astype(int)
    iou = jaccard_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)
    return {'IoU': iou, 'F1': f1}
