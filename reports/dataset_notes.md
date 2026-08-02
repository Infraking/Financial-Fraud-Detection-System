# PaySim Dataset Notes

## 📊 Dataset Overview
- **Name:** PaySim Synthetic Financial Dataset
- **Source:** Kaggle - "Synthetic Financial Datasets For Fraud Detection"
- **Size:** ~6.3 million rows
- **Time Period:** 31 days of simulated mobile money transactions (step = 1 hour)

---

## 📋 Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Time step (1 hour = 1 step), ranges from 1 to 744 (31 days) |
| `type` | string | Transaction type: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN |
| `amount` | float | Transaction amount in local currency |
| `nameOrig` | string | Customer ID who initiated the transaction |
| `oldbalanceOrig` | float | Sender's balance **BEFORE** the transaction |
| `newbalanceOrig` | float | Sender's balance **AFTER** the transaction |
| `nameDest` | string | Customer ID who received the transaction |
| `oldbalanceDest` | float | Receiver's balance **BEFORE** the transaction |
| `newbalanceDest` | float | Receiver's balance **AFTER** the transaction |
| `isFraud` | int | **TARGET VARIABLE:** 1 = Fraud, 0 = Legitimate |
| `isFlaggedFraud` | int | PaySim's naive rule: 1 if TRANSFER > 200,000 (baseline) |

---

## 🔍 Key Insights for Feature Engineering

### 1. Fraud Distribution
- **Fraud is EXTREMELY rare:** Only ~0.1-0.3% of transactions are fraudulent
- This is why **accuracy is a misleading metric** - a model that predicts "legitimate" 100% of the time is 99.8% accurate but catches ZERO fraud

### 2. Fraud Only Happens in Two Transaction Types
```python
# Run this in your EDA
df.groupby('type')['isFraud'].sum()
