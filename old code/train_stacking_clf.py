from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.metrics import auc, roc_curve
import h5py
import numpy as np
import pickle
from tqdm import tqdm
import warnings
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingClassifier


warnings.filterwarnings('ignore')

with h5py.File('./mfcc_feature.h5', 'r') as h5_data:
    data = np.array(h5_data['dataset'])

with h5py.File('./label_mfcc.h5', 'r') as h5_data:
    label = np.array(h5_data['dataset'])

sc = pickle.load(open('./scaler.pkl', 'rb'))
data = sc.fit_transform(data)

# SMOTE oversampling
data, label = SMOTE().fit_resample(data, label)

# pickle.dump(sc, open('./scaler.pkl', 'wb'))

stf = StratifiedKFold(n_splits=10, shuffle=True)

rfc = pickle.load(open('./rfc_model_smote.pkl', 'rb'))

lgbm = pickle.load(open('./lgbm_auc_smote.pkl', 'rb'))

lr = RidgeClassifier()

lg = LGBMClassifier()

# =========================================================================== #
# # Lightgbm meta learner

# # AUC 0.93797

stc_score = []
stc = StackingClassifier(classifiers=[rfc, lgbm],
                         meta_classifier=lg, fit_base_estimators=False)

for train_index, test_index in tqdm(stf.split(data, label), total=stf.get_n_splits(),
                                    desc='stratified k-fold stacking clasifier', colour='green'):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    stc.fit(X_train, y_train)
    predictions_stc = stc.predict(X_test)
    fpr_stc, tpr_stc, thresholds_stc = roc_curve(y_test, predictions_stc, pos_label=1)
    score_auc_stc = auc(fpr_stc, tpr_stc)
    stc_score.append(score_auc_stc)

# print(np.array(stc_score).mean())
# pickle.dump(stc, open('./stacking_model_lightgbm_smote.pkl', 'wb'))

# ======================================================================== #
# # Catboost meta learner

# # AUC 0.93716

# stc_score = []
# stc = StackingClassifier(classifiers=[rfc, lgbm],
#                           meta_classifier=ct, fit_base_estimators=False)
#
#
# for train_index, test_index in tqdm(stf.split(data, label), total=stf.get_n_splits(), desc='stratified k-fold stacking clasifier', colour='green'):
#     X_train, X_test = data[train_index], data[test_index]
#     y_train, y_test = label[train_index], label[test_index]
#     stc.fit(X_train, y_train)
#     predictions_stc = stc.predict(X_test)
#     fpr_stc, tpr_stc, thresholds_stc = roc_curve(y_test, predictions_stc, pos_label=1)
#     score_auc_stc = auc(fpr_stc, tpr_stc)
#     stc_score.append(score_auc_stc)
#
#
# print(np.array(stc_score).mean())
# pickle.dump(stc, open('./stacking_model_catboost_smote.pkl', 'wb'))

# stkc = pickle.load(open('./stacking_model_lightgbm_smote.pkl', 'rb'))


## Training base model old code

# with h5py.File('./mfcc_feature.h5', 'r') as h5_data:
#     data = np.array(h5_data['dataset'])
#
# with h5py.File('./label_mfcc.h5', 'r') as h5_data:
#     label = np.array(h5_data['dataset'])
#
#
# # Scaling
# sc = StandardScaler()
# data = sc.fit_transform(data)
#
# # SMOTE oversampling
# data, label = SMOTE().fit_resample(data, label)
#
# # pickle.dump(sc, open('./scaler.pkl', 'wb'))
#
# stf = StratifiedKFold(n_splits=10, shuffle=True)
#
# # ================================================================================================== #
#
# # # AUC rfc 91.49%
#
# rfc_score = []
# rfc = RandomForestClassifier(n_jobs=-1)
# for train_index, test_index in tqdm(stf.split(data, label), total=stf.get_n_splits(), desc='stratified k-fold rfc', colour='green'):
#     X_train, X_test = data[train_index], data[test_index]
#     y_train, y_test = label[train_index], label[test_index]
#     rfc.fit(X_train, y_train)
#     predictions_rfc = rfc.predict(X_test)
#     fpr, tpr, thresholds = roc_curve(y_test, predictions_rfc, pos_label=1)
#     score_auc_rfc = auc(fpr, tpr)
#     rfc_score.append(score_auc_rfc)
#
# print(np.array(rfc_score).mean())
# pickle.dump(rfc, open('./rfc_model_smote.pkl', 'wb'))
#
#
# # ===================================================================================== #
#
# # # Light gbm Classifier
# # # AUC 90.32%
# #
# score_lgbm = []
# lgbm = LGBMClassifier(n_jobs=-1)
# for train_index, test_index in tqdm(stf.split(data, label), total=stf.get_n_splits(), desc='stratified k-fold lightgbm',
#                                     colour='green'):
#     X_train, X_test = data[train_index], data[test_index]
#     y_train, y_test = label[train_index], label[test_index]
#     lgbm.fit(X_train, y_train, eval_metric='auc')
#     predictions_lgbm = lgbm.predict(X_test)
#     fpr_lgbm, tpr_lgbm, thresholds_xg = roc_curve(y_test, predictions_lgbm, pos_label=1)
#     score_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
#     score_lgbm.append(score_auc_lgbm)
#
# print(np.array(score_lgbm).mean())
# pickle.dump(lgbm, open('./lgbm_auc_smote.pkl', 'wb'))
