#! /usr/bin/env python
'''
:mod: classify -- Classification methods
========================================
..module:: classification
  :platform: Unix
  :synopsis: Classify a string as english or Hinglish
..moduleauthor:: Tushar Dhyani

Requirements::
	1. You will need to install keras
	2. You will need the pretrained Word-Embeddings

'''

import keras
import json
import os
import logging
import numpy as np

logging.basicConfig()
LOG = logging.getLogger('classifier')
LOG.setLevel(logging.INFO)

class classifier(object):
	'''Classifies a text into Hinglish and English'''
	

	def __init__(self, document=None):
		'''
		'''
		self.document = document
		with open('we.json','r') as f:
			self._embeddings = json.load(f)
		with open('mapping.json','r') as f:
			self._tokens = json.load(f)
		self._t = keras.preprocessing.text.Tokenizer()
		self._t.fit_on_texts(list(sorted(self._embeddings.keys())))

		self.embedding_matrix = np.zeros((len(self._t.word_index) + 1, 50))
		for word, i in self._t.word_index.items():
		    embedding_vector = embeddings_index.get(word)
		    if embedding_vector is not None:
		        self.embedding_matrix[i] = embedding_vector

		# self.embedding_vector = np.zeros(())
		# with open('model.json','r') as f:
		# 	self.model = f.read()
		# self.model = keras.models.model_from_json(self.model)
		# self.model.load_weights('model.h5')

	@classmethod
	def preprocess_string(cls, document=None):
		'''
		Preprocess the text and 
		replace words with their embeddings.

		Args:
		document(str) : The string to be processed.
		'''
		return cls(document.lower())

	def predict(self):
		'''
		Predict the value as 1: Hinglish and 0: English
		Pass the values to the class using the classifier.preprocess_string(string)
		and then generate the output on the string.
		'''
		seq = keras.preprocessing.text.text_to_word_sequence(self.document)
		processed_seq=[self._tokens[i]+1 for i in seq if i in list(self._tokens.keys())]
		padded_seq = np.zeros((self._t.document_count,50), dtype='int32')
		for i,word_loc in enumerate(processed_seq[:50]):
			padded_seq[word_loc,i]=1

		return len(padded_seq)