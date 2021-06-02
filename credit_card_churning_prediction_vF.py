import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import csv file, preprocessing through dummy variables, create X and y datasets
dataset = pd.read_csv('BankChurners.csv')
pd.set_option('display.max_columns', None)
# dropping last 2 columns
dataset = dataset.iloc[:,:-2]
# forming dummy variables for categorical columns
dataset2 = pd.get_dummies(dataset, columns=['Gender','Education_Level', 'Marital_Status','Income_Category',
                                  'Card_Category'])
dataset2.head()

X = dataset2.iloc[:,2:].values
y = dataset2.iloc[:,1].values

# Data preprocessing by label encoding for y outcomes
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = 1-y
y

# Split X and y datasets into testing and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,:13] = sc.fit_transform(X_train[:,:13])
X_test[:,:13] = sc.transform(X_test[:,:13])
print(X_train)
print(X_test)

# Trying Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state = 0)
classifier_logreg.fit(X_train, y_train)
# Predicting the Test set results with Logistic Regression
y_pred_logreg = classifier_logreg.predict(X_test)

# Trying K-NN model
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)
# Predicting the Test set results with K-NN
y_pred_knn = classifier_knn.predict(X_test)

# Trying the Kernel SVM model
from sklearn.svm import SVC
classifier_kernelsvm = SVC(kernel = 'rbf', random_state = 0)
classifier_kernelsvm.fit(X_train, y_train)
# Predicting the Test set results with Kernel SVM
y_pred_kernelsvm = classifier_kernelsvm.predict(X_test)

# Trying the Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_nb = classifier_nb.predict(X_test)

# Trying the Random Forest Classification model
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)
# Predicting the Test set results
y_pred_rf = classifier_rf.predict(X_test)

# Trying the ANN model
import tensorflow as tf
#tf.__version__
import scipy as sc
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
# Initializing the ANN
ann_classifier = tf.keras.Sequential()
# Adding the input layer and the first and second hidden layer
ann_classifier.add(tf.keras.layers.Dense(units=12, activation='relu'))
ann_classifier.add(tf.keras.layers.Dense(units=8, activation='relu'))
# Adding the output layer
ann_classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# Compiling the ANN
ann_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics='AUC')
# Training the ANN on the Training set
ann_classifier.fit(x=X_train, y=y_train, batch_size=30, epochs=200)
# Predicting the Test set results
y_pred_ann = ann_classifier.predict(x=X_test)
y_pred_ann_2 = (y_pred_ann > 0.5)

# For K-Fold Cross Validation and Random Search CV, we need KerasClassifier, 
# which requires a function that creates the model
# Let's write that function:
def create_model(optimizer='adam', loss='binary_crossentropy', metrics='AUC'):
	# create model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(37,)))
    model.add(tf.keras.layers.Dense(12, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	# Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
# create KerasClassifier wrapper for use with scikit learn
ann_classifier_wrapper_kfold = KerasClassifier(build_fn=create_model, epochs=200, batch_size=30, verbose=0)
ann_classifier_wrapper_grid = KerasClassifier(build_fn=lambda optimizer, loss, metrics: create_model(optimizer, loss, metrics), epochs=200, batch_size=30, verbose=0)

# Trying XGBoost
from xgboost import XGBClassifier
classifier_xgb = XGBClassifier(n_estimators=30, max_depth=6, use_label_encoder = False)
classifier_xgb.fit(X_train, y_train, eval_metric='logloss')
# only if want to see parameters
# classifier_xgb.get_booster
# Predicting the Test set results
y_pred_xgb = classifier_xgb.predict(X_test)

# Trying CatBoost
from catboost import CatBoostClassifier
classifier_cb = CatBoostClassifier(iterations=100)
classifier_cb.fit(X_train, y_train)
# Predicting the Test set results
y_pred_cb = classifier_cb.predict(X_test)
classifier_cb.get_all_params()

# Making the Confusion Matrices
from sklearn.metrics import confusion_matrix, accuracy_score
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_kernelsvm = confusion_matrix(y_test, y_pred_kernelsvm)
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_ann = confusion_matrix(y_test, y_pred_ann_2)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_cb = confusion_matrix(y_test, y_pred_cb)

# Making the K-Fold Cross-Validation scores for ROC AUC
from sklearn.model_selection import cross_val_score
# create StatifiedKFold to use for each model's score
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accuracies_logreg_roc = cross_val_score(estimator = classifier_logreg, X = X, y = y, cv = kfold, scoring='roc_auc')
accuracies_knn_roc = cross_val_score(estimator = classifier_knn, X = X, y = y, cv = kfold, scoring='roc_auc')
accuracies_kernelsvm_roc = cross_val_score(estimator = classifier_kernelsvm, X = X, y = y, cv = kfold, scoring='roc_auc')
accuracies_nb_roc = cross_val_score(estimator = classifier_nb, X = X, y = y, cv = kfold, scoring='roc_auc')
accuracies_rf_roc = cross_val_score(estimator = classifier_rf, X = X, y = y, cv = kfold, scoring='roc_auc')
accuracies_ann_roc = cross_val_score(estimator = ann_classifier_wrapper_kfold, X = X, y = y, cv = kfold, scoring='roc_auc')
accuracies_xgb_roc = cross_val_score(estimator = classifier_xgb, X = X, y = y, cv = kfold, scoring='roc_auc')
accuracies_cb_roc = cross_val_score(estimator = classifier_cb, X = X, y = y, cv = kfold, scoring='roc_auc')

# Making the K-Fold Cross-Validation scores for Accuracy
accuracies_logreg_accuracy = cross_val_score(estimator = classifier_logreg, X = X, y = y, cv = kfold, scoring = 'accuracy')
accuracies_knn_accuracy = cross_val_score(estimator = classifier_knn, X = X, y = y, cv = kfold, scoring = 'accuracy')
accuracies_kernelsvm_accuracy = cross_val_score(estimator = classifier_kernelsvm, X = X, y = y, cv = kfold, scoring = 'accuracy')
accuracies_nb_accuracy = cross_val_score(estimator = classifier_nb, X = X, y = y, cv = kfold, scoring = 'accuracy')
accuracies_rf_accuracy = cross_val_score(estimator = classifier_rf, X = X, y = y, cv = kfold, scoring = 'accuracy')
accuracies_ann_accuracy = cross_val_score(estimator = ann_classifier_wrapper_kfold, X = X, y = y, cv = kfold, scoring = 'accuracy')
accuracies_xgb_accuracy = cross_val_score(estimator = classifier_xgb, X = X, y = y, cv = kfold, scoring = 'accuracy')
accuracies_cb_accuracy = cross_val_score(estimator = classifier_cb, X = X, y = y, cv = kfold, scoring = 'accuracy')

# Printing results
print('CONFUSION MATRIX'
      '\nLogistic Regression: \n', cm_logreg,
      '\nK-NN: \n', cm_knn,
      '\nKernel SVM: \n', cm_kernelsvm,
      '\nNaive Bayes: \n', cm_nb,
      '\nRandom Forest: \n', cm_rf,
      '\nArtifical Neural Network: \n', cm_ann,
      '\nXGBoost: \n', cm_xgb,
      '\nCatBoost: \n', cm_cb)

print('K-FOLD CROSS-VALIDATION SCORES (ROC AUC)'
      '\n   ' 'Logistic Regression: ', accuracies_logreg_roc.mean()*100,
      '\n   ' 'K-NN: ', accuracies_knn_roc.mean()*100,
      '\n   ' 'Kernel SVM: ', accuracies_kernelsvm_roc.mean()*100,
      '\n   ' 'Naive Bayes: ', accuracies_nb_roc.mean()*100,
      '\n   ' 'Random Forest: ', accuracies_rf_roc.mean()*100,     
      '\n   ' 'Artificial Neural Network: ', accuracies_ann_roc.mean()*100,
      '\n   ' 'XGBoost: ', accuracies_xgb_roc.mean()*100,
      '\n   ' 'CatBoost: ', accuracies_cb_roc.mean()*100)

print('K-FOLD CROSS-VALIDATION SCORES (Accuracy)'
      '\n   ' 'Logistic Regression: ', accuracies_logreg_accuracy.mean()*100,
      '\n   ' 'K-NN: ', accuracies_knn_accuracy.mean()*100,
      '\n   ' 'Kernel SVM: ', accuracies_kernelsvm_accuracy.mean()*100,
      '\n   ' 'Naive Bayes: ', accuracies_nb_accuracy.mean()*100,
      '\n   ' 'Random Forest: ', accuracies_rf_accuracy.mean()*100,     
      '\n   ' 'Artificial Neural Network: ', accuracies_ann_accuracy.mean()*100,
      '\n   ' 'XGBoost: ', accuracies_xgb_accuracy.mean()*100,
      '\n   ' 'CatBoost: ', accuracies_cb_accuracy.mean()*100)


# CatBoost is the top performing model (in terms of K-Fold Cross Validation), so let's optimize parameters via Optuna Library
from sklearn.model_selection import cross_validate
from optuna import create_study
# Other arguments can include in parameters - 'loss_function': ['CrossEntropy', 'logloss'], 'custom_metric': ['Logloss', 'CrossEntropy', 'AUC'], 'bootstrap_type':['Bayesian','Bernoulli', 'MVS', 'Poisson']
def catboost_objective(trial):
    iterations = int(trial.suggest_loguniform('iterations', 100, 300))
    depth = int(trial.suggest_loguniform('depth', 1, 50))
    eval_metric = trial.suggest_categorical('eval_metric', ['Logloss', 'CrossEntropy', 'AUC'])
    #bootstrap_type = trial.suggest_categorical('bootstrap_type', ['MVS'])
    model = CatBoostClassifier(iterations=iterations, 
                          depth=depth,
                          eval_metric = eval_metric
                          #bootstrap_type = bootstrap_type
                          )
    scores = cross_validate(model, X, y, cv=kfold, scoring='roc_auc')
    return scores['test_score'].mean()
study = create_study(direction='maximize')
study.optimize(catboost_objective, n_trials=20)
best_trial =  study.best_trial
print('AUC: {}'.format(best_trial.value))
print('Best params: {}'.format(best_trial.params))


# Since XGBoost is the 2nd best performing model, we also apply Optuna
def xgb_objective(trial):
    n_estimators = int(trial.suggest_loguniform('n_estimators', 100, 400))
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 50))
    booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
    tree_method = trial.suggest_categorical('tree_method', ['exact','approx','hist','gpu_hist'])
    #eval_metric = trial.suggest_categorical('eval_metric', ['Logloss', 'AUC'])
    model = XGBClassifier(n_estimators=n_estimators, 
                          max_depth=max_depth,
                          booster = booster,
                          tree_method = tree_method
                          #bootstrap_type = bootstrap_type
                          )
    scores = cross_validate(model, X, y, cv=kfold, scoring='roc_auc')
    return scores['test_score'].mean()
study = create_study(direction='maximize')
study.optimize(xgb_objective, n_trials=20)
best_trial =  study.best_trial
print('AUC: {}'.format(best_trial.value))
print('Best params: {}'.format(best_trial.params))


# Since Random Forest is the 3rd best performing model, let's also Optuna
def rf_objective(trial):
    n_estimators = int(trial.suggest_loguniform('n_estimators', 100, 400))
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 50))
    criterion = trial.suggest_categorical('criterion', ['entropy'])
    model = RandomForestClassifier(n_estimators=n_estimators, 
                          max_depth=max_depth,
                          criterion = criterion
                          )
    scores = cross_validate(model, X, y, cv=kfold, scoring='roc_auc')
    return scores['test_score'].mean()
study = create_study(direction='maximize')
study.optimize(rf_objective, n_trials=20)
best_trial =  study.best_trial
print('AUC: {}'.format(best_trial.value))
print('Best params: {}'.format(best_trial.params))


# ANN could perform significantly better if we optimize it, so let's apply Random Search CV (since Optuna is not compatible with Keras)
from sklearn.model_selection import RandomizedSearchCV
parameters_ann = [{'epochs': [200], 'batch_size': [30, 35, 40, 45, 50], 'optimizer': ['adam'], 'loss': ['mse','binary_crossentropy'], 'metrics': ['AUC','accuracy']}]
ann = RandomizedSearchCV(ann_classifier_wrapper_grid, parameters_ann, n_iter=30,
                                        scoring="roc_auc", cv=kfold,
                                        n_jobs=-1)
ann.fit(X, y)
best_accuracy = ann.best_score_
best_param = ann.best_params_
print('AUC: {:.2f} %'.format(best_accuracy*100))
print('Best params:{}'.format(best_param))


# FINAL WINNER: CATBOOST yields the best score of ~99.46%, with optimal parameters as depth of ~5.04, eval_metric of AUC, iterations of ~259.60

