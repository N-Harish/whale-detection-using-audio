import pickle
from feature_gen import *


test_feature('./features/test_features.h5', './whale_data/data/test_wav')

with open('./models/scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

with open('./models/stacking_model_lightgbm_smote.pkl', 'rb') as f:
    model = pickle.load(f)

with h5py.File('./features/test_features.h5', 'r') as f:
    data = np.array(f['dataset'])

data = sc.transform(data)

pred = model.predict(data)

df = pd.DataFrame(pred)
df.to_csv('./submission.csv', header=False, index=False)
