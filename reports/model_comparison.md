# Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Notes |
|---------|---------|---------|---------|---------|---------|---------|
| Logistic Regression |0.778|0.640|0.703|0.996|0.787|The decision threshold was increased to 0.99 to improve the balance between precision and recall. This increased the F1 score from 0.640 to 0.703 while maintaining excellent ROC-AUC performance. |
| Random Forest |1.0| 0.740|0.851|0.992|0.850|Trained on a representative subset of the dataset to improve computational efficiency. The model demonstrated strong fraud detection performance with high ROC-AUC and PR-AUC scores while maintaining a good balance between precision and recall. |
| XGBoost |0.991|0.998|0.994|0.999|0.998|Best performing model in the project, Achieved very high fraud detection performance with minimal false positives and false negatives.|
| Isolation Forest | | | | | | |


### Conclusion

XGBoost was selected as the final deployment model because it achieved the highest Precision, Recall, F1 Score, ROC-AUC, and PR-AUC among all evaluated models.

### Top Features

| Feature | Importance |
|----------|----------|
| errorBalanceOrig | 0.4907 |
| origEmptied | 0.2576 |
| oldbalanceOrg | 0.0923 |
| newbalanceOrig | 0.0780 |
| amount | 0.0340 |

### Interpretation

- Balance inconsistencies were the strongest indicator of fraud.
- Transactions that emptied the sender account were highly associated with fraudulent behavior.
- Transaction amount and account balance patterns played an important role in model decisions.
