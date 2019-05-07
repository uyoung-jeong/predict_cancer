import numpy as np
from xgboost import XGBClassifier, Booster

import pickle
import ipdb
import tqdm
from data_loader import Dataset
from sklearn.model_selection import train_test_split
import random
import os

# read and preprocess data
dataset = Dataset()

# split data into train and valid set
patho = dataset.patho_data
patho_keys = list(patho.keys())
neutral = dataset.neutral_data
neutral_keys = list(neutral.keys())

x_patho = []
for k in patho_keys:
	x_patho.append(patho[k][1:])

x_neutral = []
for k in neutral_keys:
	x_neutral.append(neutral[k][1:])

#x_patho = np.asarray([e[-1] for e in patho])
#x_neutral = np.asarray([e[-1] for e in neutral])
x = np.concatenate((x_patho, x_neutral), axis=0)
start = np.array(list(x[:,0]), dtype=np.float)
end = np.array(list(x[:,1]), dtype=np.float)
seq = np.array(list(x[:,2]), dtype=np.float)
x = np.concatenate((start, end, seq), axis=1)

print('pathogenic data: ' + str(len(x_patho)))
print('neutral data: ' + str(len(x_neutral)))
y = np.ones(len(x_patho))
y = np.concatenate((y, np.zeros(len(x_neutral))), axis=0)
print('splitting data into train and validation set')

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
print('training set size: ' + len(y_train))
print('validation set size: ' + len(y_valid))

ipdb.set_trace()

# build model
evallist = [(x_valid, y_valid)]
model = XGBClassifier()
if not os.path.isfile('model.bin'):
	print('start training')
	model.fit(x_train, y_train, eval_set=evallist, verbose=True, early_stopping_rounds=20)

	print('validation result')
	print(model.score(x_valid, y_valid))
	model.save_model('model.bin')

else:
	booster = Booster()
	booster.load_model('model.bin')
	model._Booster = booster
"""
print('check Nematostella vectensis seq')
nv_seq = "TSPDIMSSSFYIDSLISKAKSVPTSTSEPRHTYESPVPCSCCWTPTQPDPSSLCQLCIPTSASVHPYMHHVRGASIPSGAGLYSRELQKDHILLQQHYAATEEERLHLASYASSRDPDSPSRGGNSRSKRIRTAYTSMQLLELEKEFSQNRYLSRLRRIQIAALLDLSEKQVKIWFQNRRVKWKKDKKAAQHGTTTETSSCPSSPASTGRMDGV"
nv_vec = dataset.one_hot(nv_seq)
predict_data = np.asarray([nv_vec]*2)
prediction = model.predict_proba(predict_data)
print('prediction: ')
print(prediction)
"""
ipdb.set_trace()
print('finished execution')
