import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV

kappa_scorer = make_scorer(cohen_kappa_score)
scoring = {
    'accuracy': 'accuracy',
    'kappa': kappa_scorer
}

def grid_search(X_train, y_train, param_grid, cv=3, scoring=scoring):
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=10,
        eval_metric='mlogloss',
        tree_method='hist',
        early_stopping_rounds=50,
        random_state=42,
        )
    grid = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit='accuracy',  
            verbose=1,
            n_jobs=-1
        )
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_, grid.best_estimator_

params = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5],
    }

dtrain = np.load("X_train_scaled.npy")
dtest = np.load("X_test_scaled.npy")
ytrain = np.load("y_train.npy")
ytest = np.load("y_test.npy")

print(grid_search(dtrain, ytrain, params, cv=3, scoring=scoring))
