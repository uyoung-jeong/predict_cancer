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
print('training set size: ' + str(len(y_train)))
print('validation set size: ' + str(len(y_valid)))

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

print('check cancer cases')
brca1 = open(os.path.join('data', 'brca1.txt'), 'r').readlines()
brca1 = ''.join(brca1)
brca1 = brca1.replace('\n', '')

brca2 = open(os.path.join('data', 'brca2.txt'), 'r').readlines()
brca2 = ''.join(brca2)
brca2 = brca2.replace('\n', '')

brca1_vec = dataset.one_hot(brca1)
brca2_vec = dataset.one_hot(brca2)

predict_data = np.array([[dataset.one_hot('S', max_len=1), dataset.one_hot('F', max_len=1), brca1_vec], # S4F, BC, passenger
						 [dataset.one_hot('V', max_len=1), dataset.one_hot('M', max_len=1), brca1_vec], # V271M, BC, passenger
						 [dataset.one_hot('P', max_len=1), dataset.one_hot('S', max_len=1), brca1_vec], # P346S, BC, passenger
						 [dataset.one_hot('T', max_len=1), dataset.one_hot('M', max_len=1), brca1_vec], # T231M, BC, passenger
						 [dataset.one_hot('F', max_len=1), dataset.one_hot('L', max_len=1), brca2_vec], # F32L, BC, passenger
						 [dataset.one_hot('Y', max_len=1), dataset.one_hot('C', max_len=1), brca2_vec], # Y42C, BC, passenger
						 [dataset.one_hot('T', max_len=1), dataset.one_hot('I', max_len=1), brca2_vec], # T64I, BC, passenger
						 [dataset.one_hot('K', max_len=1), dataset.one_hot('R', max_len=1), brca2_vec], # K53R, BC, passenger
						 [dataset.one_hot('I', max_len=1), dataset.one_hot('L', max_len=1), brca2_vec], # I982L, Normal, passenger
						 [dataset.one_hot('M', max_len=1), dataset.one_hot('V', max_len=1), brca2_vec], # M784V, Normal, passenger
						])

start = np.array(list(predict_data[:,0]), dtype=np.float)
end = np.array(list(predict_data[:,1]), dtype=np.float)
seq = np.array(list(predict_data[:,2]), dtype=np.float)
predict_data = np.concatenate((start, end, seq), axis=1)

prediction = model.predict_proba(predict_data)
print('prediction: ')
print(prediction)
ipdb.set_trace()
print('finished execution')
