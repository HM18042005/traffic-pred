# Demand Prediction - CatBoost Report

This report summarises the single-model pipeline used to predict the `demand` column. Validation scores are 5-fold out-of-fold (OOF) R².

## 1. Final model summary

- Final OOF R²: **0.99575**
- OOF mean R²: **0.99575**
- OOF std R²: **0.00033**
- Optuna best R²: **0.99577**
- Optuna trials completed: **10**

## 2. Best parameters

- bagging_temperature: `0.9488855372533332`
- border_count: `247`
- depth: `5`
- iterations: `2267`
- l2_leaf_reg: `4.050837781329674`
- learning_rate: `0.03912141628549695`
- random_strength: `0.3252579649263976`

## 3. Fold scores

0.99567, 0.99609, 0.99573, 0.99520, 0.99608

## 4. Feature engineering

The retained feature set includes timestamp-derived cyclical features, geohash prefix hierarchy, geohash latitude/longitude decoding, geohash statistics, per-(geohash, hour) statistics, day-48 lookup features, interaction features, and out-of-fold target encoding for geohash and the other high-signal categorical fields.

- Engineered feature count: **43**

## 5. Submission

`outputs/submission.csv` contains clipped predictions for the full test set (41778 rows).
