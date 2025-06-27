import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer, classification_report
from sklearn.model_selection import GridSearchCV

kappa_scorer = make_scorer(cohen_kappa_score)
scoring = {
    'accuracy': 'accuracy',
    'kappa': kappa_scorer
}

def grid_search(X_train, y_train, param_grid, cv=2, scoring=scoring):
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=10,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        error_score='raise',
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
        'n_estimators': [400],
        'max_depth': [9],
        'learning_rate': [0.1],
        'subsample': [0.7],
        'colsample_bytree': [0.5],
        'gamma': [0.2],
    }

dtrain = np.load("X_train_scaled.npy")
dtest = np.load("X_test_scaled.npy")
ytrain = np.load("y_train.npy")
ytest = np.load("y_test.npy")

# best_params, _, best_estimator = grid_search(dtrain, ytrain, params, cv=2, scoring=scoring)
best_estimator = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=9,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.5,
    gamma=0.2,
    objective='multi:softmax',
    num_class=10,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42
)
ypred = best_estimator.predict(dtest)
print("Best parameters:", best_params)
print(classification_report(ytest, ypred))
print(cohen_kappa_score(ytest, ypred))
