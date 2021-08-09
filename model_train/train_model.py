from numpy import ndarray
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
import h5py
import numpy as np
import pickle
from tqdm import tqdm
import warnings
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import RidgeClassifier


warnings.filterwarnings('ignore')


def load_data(feature_path: str, label_path: str) -> (ndarray, ndarray):
    """

    :param label_path: path of label file
    :param feature_path: path of feature file
    :type feature_path: str
    :type label_path: str
    """
    with h5py.File(feature_path, 'r') as h5_dt:
        dt: ndarray = np.array(h5_dt['dataset'])

    with h5py.File(label_path, 'r') as h5_lbl:
        lbl: ndarray = np.array(h5_lbl['dataset'])

    return dt, lbl


def train_model(ft_path: str, lb_path: str, model_path: str, scaler_path: str):
    """

    :param ft_path: path to feature file
    :param lb_path: path to label file
    :param model_path: path to save model
    :param scaler_path:path to save scaler
    :type ft_path: object
    :type lb_path: object
    :type model_path: str
    :type scaler_path: object
    """

    data, label = load_data(ft_path, lb_path)

    sc = StandardScaler()

    data = sc.fit_transform(data)

    # SMOTE oversampling
    data, label = SMOTE().fit_resample(data, label)

    # pickle.dump(sc, open('./scaler.pkl', 'wb'))

    stf = StratifiedKFold(n_splits=10, shuffle=True)

    # ================================================================================================== #

    # # AUC rfc 91.49%

    rfc_score = []
    rfc = RandomForestClassifier(n_jobs=-1)
    for train_index, test_index in tqdm(stf.split(data, label), total=stf.get_n_splits(), desc='stratified k-fold rfc',
                                        colour='green'):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        rfc.fit(X_train, y_train)
        predictions_rfc = rfc.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predictions_rfc, pos_label=1)
        score_auc_rfc = auc(fpr, tpr)
        rfc_score.append(score_auc_rfc)

    print(np.array(rfc_score).mean())

    # ===================================================================================== #

    # # Light gbm Classifier
    # # AUC 90.32%
    #
    score_lgbm = []
    lgbm = LGBMClassifier(n_jobs=-1)
    for train_index, test_index in tqdm(stf.split(data, label), total=stf.get_n_splits(),
                                        desc='stratified k-fold lightgbm',
                                        colour='green'):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        lgbm.fit(X_train, y_train, eval_metric='auc')
        predictions_lgbm = lgbm.predict(X_test)
        fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(y_test, predictions_lgbm, pos_label=1)
        score_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
        score_lgbm.append(score_auc_lgbm)

    print(np.array(score_lgbm).mean())

    # ================================================================================================== #
    # (lightgbm + random forest) base estimator + Lightgbm metamodel
    # AUC 0.93797

    lg = LGBMClassifier()
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

    with open(model_path, 'wb') as f:
        pickle.dump(stc, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(sc, f)
