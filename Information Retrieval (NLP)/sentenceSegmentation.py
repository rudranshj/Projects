from util import *

import re
import nltk
# nltk.download('punkt')



class SentenceSegmentation():

	def naive(self, text):


		segmentedText = None

		segmentedText=re.split('[!?.]', text)

		return segmentedText





	def punkt(self, text):

		segmentedText = None

		segmentedText=nltk.sent_tokenize(text)
		
		return segmentedText


