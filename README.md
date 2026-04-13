# Profitable Orders Driver Analysis with Logistic Regression

Uses statsmodels Logit to identify which order and customer attributes drive order profitability. The model supports the sales team in making data-informed decisions about which orders to prioritise or accept.

## Business Context

Not all orders are equally profitable. Understanding the factors that predict a profitable order allows the sales team to steer their effort and discount behaviour more strategically, improving the overall margin of the order book.

## Dataset

`profitable_orders.csv` contains order-level records including `profit`, `discount_rate`, `items_per_order`, `average_item_value`, demographic fields (`gender`, `age_group`), and `new_customer` and `loyalty_program` flags.

## Methodology

**Target Variable:** `profit_binary` is 1 when profit > 0, otherwise 0.

**EDA:** Histograms for discount_rate, average_item_value, and items_per_order. The dataset is naturally imbalanced (~66% profitable orders).

**Outlier Removal:** Records with items_per_order >= 10 and average_item_value >= 250 are removed.

**Preprocessing:** Categorical variables (gender, age_group) are one-hot encoded with `drop_first=True`. A constant is added to X via `sm.add_constant`. 80/20 train/test split with `random_state=1502`.

**Model:** statsmodels `Logit` — chosen over sklearn for its interpretable summary output including coefficients, p-values, and confidence intervals.

**Coefficient Interpretation:** Coefficients are converted to percentage probability changes using the formula `(exp(β) - 1) × 100%`.

**Evaluation:** Accuracy, Sensitivity (Recall), Specificity, F1 Score, and a confusion matrix plot.

## Project Structure

```
10_profitable_orders_logistic_regression/
├── profitable_orders.py  # Full analysis pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
statsmodels
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `profitable_orders.csv` in the same directory and run:

```bash
python profitable_orders.py
```

Outputs: `distributions_before.png`, `confusion_matrix.png`, printed model summary and metrics.
