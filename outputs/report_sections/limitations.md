# Limitations

## Dataset bias
The IBM Telco Customer Churn dataset is a **synthetic, single-domain snapshot** designed for teaching. It may not reflect the true demographic, geographic, or behavioral mix of any real operator. Models trained here can inherit **historical or sampling bias** present in the features (e.g., contract types, payment methods) without representing causal churn drivers.

## Single-company data limitation
All records come from one fictional telecom provider. **Findings do not generalize** to other industries, countries, pricing regimes, or competitive markets. External validity is limited: a model that works on this table may fail when product bundles, regulations, or customer expectations differ.

## Temporal drift
The dataset has **no time index**. Customer behavior, macroeconomic conditions, and service offerings change over time (**concept drift**). A model trained on a static export may degrade when churn patterns shift (e.g., new fiber plans, promotional campaigns). Retraining pipelines and monitoring live performance are required for production use.

## Missing churn drivers
Important real-world predictors are absent: customer service interactions, network quality complaints, competitor offers, social sentiment, and voluntary vs. involuntary churn. **Unobserved confounders** can make reported feature importances misleading if interpreted as causes rather than correlations.

## Generalization limitations
- **Class imbalance** (~27% churn) makes accuracy an optimistic metric; threshold tuning for business costs was not fully explored.
- **Preprocessing choices** (e.g., imputing `TotalCharges` as 0 for `tenure=0`) are reasonable but affect linear models.
- **Hold-out test set** is single-split; uncertainty is reported via CV on train, but test metrics have sampling variance.
- Models optimize **generic classification metrics**, not revenue loss or retention campaign ROI.
