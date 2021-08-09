from model_train import *
from feature_gen import *


# Generate mfcc
mfcc_gen(r'./whale_data/data/train.csv',
         './whale_data/data/train_wav',
         './features/mfcc_feature.h5',
         './features/label_mfcc.h5')

# Train stacking classifier
train_model(ft_path='./features/mfcc_feature.h5',
            lb_path='./features/label_mfcc.h5',
            model_path='./models/stacking_model_lightgbm_smote.pkl',
            scaler_path='./models/scaler.pkl')
