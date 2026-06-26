import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss

# ==========================================
# 1. DATA SAMPLING & PREPARATION
# ==========================================
# Assuming df_train is your 3 million row DataFrame
# Replace 'churn' with your target column and 'offered_price' with your price column
TARGET_COL = 'churn'
PRICE_COL = 'offered_price'

# Sample 300,000 rows for faster Optuna tuning, maintaining the churn class ratio
df_sample = df_train.sample(n=300000, random_state=42, stratify=df_train[TARGET_COL])

X_sample = df_sample.drop(columns=[TARGET_COL])
y_sample = df_sample[TARGET_COL]

# Split the sample into train and validation for Optuna evaluation
X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
)

# Identify your categorical features (ensure PRICE_COL is NOT in this list)
cat_features = ['vehicle_make', 'customer_segment', 'payment_method'] # Example list

# ==========================================
# 2. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # Define a hyperparameter space optimized for generalization and calibration
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        # Shallower trees (depth 3-6) often calibrate better and avoid over-segmentation
        'depth': trial.suggest_int('depth', 3, 7), 
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        
        # Enforce economic logic: Churn probability MUST NOT decrease as price increases.
        # '1' means monotonically increasing constraint. 
        'monotone_constraints': {PRICE_COL: 1},
        
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'cat_features': cat_features,
        'verbose': False,
        'random_seed': 42
    }
    
    # Initialize the model
    model = CatBoostClassifier(**params)
    
    # Train with early stopping on the validation set to prevent overfitting
    model.fit(
        X_train_opt, y_train_opt,
        eval_set=(X_val_opt, y_val_opt),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Predict absolute probabilities for the positive class (churn)
    preds_proba = model.predict_proba(X_val_opt)[:, 1]
    
    # Calculate Log Loss (Primary driver for well-calibrated probabilities)
    loss = log_loss(y_val_opt, preds_proba)
    
    # Optional: You can monitor Brier Score or use it as your primary return metric
    # brier = brier_score_loss(y_val_opt, preds_proba)
    # trial.set_user_attr("Brier_Score", brier)
    
    return loss

# ==========================================
# 3. RUN OPTIMIZATION
# ==========================================
# Direction is 'minimize' because lower Log Loss means better calibration
study = optuna.create_study(direction='minimize', study_name='CatBoost_Price_Churn')

# Run 50 trials (adjust based on your time constraints)
study.optimize(objective, n_trials=50)

print("\n--- Best Trial ---")
print(f"Best Log Loss: {study.best_trial.value:.5f}")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

# ==========================================
# 4. TRAIN FINAL MODEL ON FULL DATASET
# ==========================================
print("\nTraining final model on full 3M dataset...")
best_params = study.best_trial.params

# Add the static parameters back into the dictionary
best_params['monotone_constraints'] = {PRICE_COL: 1}
best_params['loss_function'] = 'Logloss'
best_params['cat_features'] = cat_features
best_params['random_seed'] = 42

# Prepare full dataset X and y
X_full = df_train.drop(columns=[TARGET_COL])
y_full = df_train[TARGET_COL]

final_model = CatBoostClassifier(**best_params)

# Train the final model (no early stopping here unless you do a cross-val split)
final_model.fit(X_full, y_full, verbose=100)
