from util import *

import nltk
from nltk.tokenize import TreebankWordTokenizer



class Tokenization():

	def naive(self, text):
		tokenizedText = []
		for s in text:
			s=s.lower()
			tokenizedText.append(s.split())  

		return tokenizedText



	def pennTreeBank(self, text):		
		tokenizedText = []

		for s in text:
			s=s.lower()
			tokenizedText.append(TreebankWordTokenizer().tokenize(s))

		return tokenizedText


