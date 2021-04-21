from util import *

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')



class StopwordRemoval():

	def fromList(self, text):
		stopwordRemovedText = []
		stopWords = set(stopwords.words('english'))
		for tokenList in text:
			subList=[]
			for token in tokenList:
				if token not in stopWords:
					subList.append(token)
			stopwordRemovedText.append(subList)
					
		return stopwordRemovedText




	