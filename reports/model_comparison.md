# Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Notes |
|---------|---------|---------|---------|---------|---------|---------|
| Logistic Regression |0.778|0.640|0.703|0.996|0.787|The decision threshold was increased to 0.99 to improve the balance between precision and recall. This increased the F1 score from 0.640 to 0.703 while maintaining excellent ROC-AUC performance. |
| Random Forest |1.0| 0.740|0.851|0.992|0.850|Trained on a representative subset of the dataset to improve computational efficiency. The model demonstrated strong fraud detection performance with high ROC-AUC and PR-AUC scores while maintaining a good balance between precision and recall. |
| XGBoost | | | | | | |
| Isolation Forest | | | | | | |
