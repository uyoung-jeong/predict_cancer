import numpy as np
from xgboost import XGBClassifier

import pickle
import ipdb
from tqdm import tqdm
import os
import random
import re

class Dataset():
	def __init__(self):
		self.patho_file = os.path.join(os.getcwd(), 'data', 'CanProVar.fa')
		self.neutral_file = os.path.join(os.getcwd(), 'data', 'humsavar_edited.txt')
		self.uniprot_file = os.path.join(os.getcwd(), 'data', 'uniprot_sprot.fasta')

		self.uniprot_pickle_path = os.path.join(os.getcwd(), 'data', 'uniprot_data.pkl')
		self.patho_pickle_path = os.path.join(os.getcwd(), 'data', 'patho_x_data.pkl')
		self.neutral_pickle_path = os.path.join(os.getcwd(), 'data', 'neutral_x_data.pkl')
		self.acid_codes = ['PAD', 'A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', 'X']
		self.idx_dict = {}
		for i, c in enumerate(self.acid_codes):
			self.idx_dict[c] = i
		self.max_len = 999
		self.X = []
		self.num_class = 2

		self.label_sheet = np.eye(self.num_class, dtype='int64')
		
		if not os.path.isfile(self.patho_pickle_path):
			print('preprocessing data')
			# read
			f = open(self.patho_file, 'r')
			patho_data = f.readlines()
			f.close()
			
			f = open(self.neutral_file, 'r')
			neutral_data = f.readlines()
			f.close()

			f = open(self.uniprot_file, 'r')
			uniprot_data = f.readlines()
			f.close()
			
			# parse
			def parse_fasta(data, is_uniprot=False):
				dic = {}
				i=0
				pbar = tqdm(total=len(data), desc='parsing fasta')
				while i < len(data):
					if data[i].find('>') != -1:
						header = data[i].replace('\n', '')
						header = data[i].replace('>', '')
						tokens = header.split()
						name = tokens[0]
						desc = ' '.join(tokens[1:])
						seq = ""
						i += 1
						pbar.update(1)
						while i < len(data) and data[i].find('>') == -1:
							seq += data[i].replace('\n', '')
							i += 1
							pbar.update(1)
						"""
						if len(seq) > self.max_len:
							self.max_len = len(seq)
						"""
						if is_uniprot: # uniprot fasta file
							if desc.find('Homo sapiens') != -1:
								keys = name.split('|')
								key = keys[1]
								dic[key] = [name, desc, seq, self.one_hot(seq)]
						else: # pathogenic fasta file; CanProVar.fa
							if desc == '':
								continue
							desc_tokens = desc.split(';')
							desc = []
							for t in desc_tokens:
								if t.find('cs') != -1:
									try:
										t = t.split(':')[1]
									except IndexError:
										print('t: ' + t)
										print('desc_tokens: ' + desc_tokens)
									match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", t, re.I)
									ori=''
									loc=9999
									end=''
									if match:
										items = match.groups()
										ori = items[0]
										loc=int(items[1])
										if loc > 999:
											continue
										end=items[2]
										dic[name] = [np.float(loc/self.max_len), 
													self.one_hot(ori, max_len=1), 
													self.one_hot(end, max_len=1), 
													self.one_hot(seq)]
					i += 1
					pbar.update(1)
				pbar.close()
				return dic
			patho_dic = parse_fasta(patho_data, False)
			uniprot_dic = parse_fasta(uniprot_data, True)
			
			# in case of neutral data, parse in differnt way
			i=0
			neutral_dic = {}
			pbar = tqdm(total=len(neutral_data), desc='parsing neutral data')
			while i < len(neutral_data):
				tokens = neutral_data[i].split()
				t3 = tokens[3].replace('p.', '')
				match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", t3, re.I)
				if not match or tokens[4].find('Polymorphism') == -1 or tokens[5] == '-':
					i += 1
					pbar.update(1)
					continue
				items = match.groups()
				ori = items[0][0]
				loc = int(items[1])
				end = items[2][0]
				try:
					neutral_dic[tokens[5]] = [np.float(loc/self.max_len), 
											 self.one_hot(ori, max_len=1), 
											 self.one_hot(end, max_len=1), 
											 uniprot_dic[tokens[1]][3]]
				except KeyError: # no entry on uniprot
					pass
				except IndexError:
					print('tokens[1]: ' + tokens[1])
					ipdb.set_trace()
					print('uniprot_dic[tokens[1]]: ' + uniprot_dic[tokens[1]])
				i += 1
				pbar.update(1)
			pbar.close()
			
			#print('max_len: ' + str(self.max_len))
			# save data
			with open(self.patho_pickle_path, 'wb') as f:
				pickle.dump(patho_dic, f, pickle.HIGHEST_PROTOCOL)
			
			with open(self.uniprot_pickle_path, 'wb') as f:
				pickle.dump(uniprot_dic, f, pickle.HIGHEST_PROTOCOL)
			
			with open(self.neutral_pickle_path, 'wb') as f:
				pickle.dump(neutral_dic, f, pickle.HIGHEST_PROTOCOL)
				
		# load data
		print('loading data')
		with open(self.patho_pickle_path, 'rb') as f:
			self.patho_data = pickle.load(f)
		with open(self.neutral_pickle_path, 'rb') as f:
			self.neutral_data = pickle.load(f)

	def one_hot(self, s, max_len = None):
		if max_len is None:
			max_len = self.max_len
		s = s.upper()
		str2vec = np.zeros((max_len,len(self.acid_codes)), dtype=np.float32)
		max_length = min(len(s), max_len)
		for i in range(max_length):
			c = s[i]
			if c in self.acid_codes:
				str2vec[i][self.idx_dict[c]] = 1
			else:
				ipdb.set_trace()
				print(c)
		return str2vec.flatten()

