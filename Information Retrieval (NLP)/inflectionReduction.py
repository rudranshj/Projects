from util import *

import nltk
import re
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')




class InflectionReduction:

	def reduce(self, text):
		

		reducedText = []
		wnl = WordNetLemmatizer()
		# Lemmatization
		for tokenList in text:
			subList=[]
			for token in tokenList:
				subList.append(wnl.lemmatize(token))
			reducedText.append(subList)
		
		return reducedText


