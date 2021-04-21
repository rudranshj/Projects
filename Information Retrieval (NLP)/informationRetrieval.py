from util import *
import math
from math import log
from math import sqrt
from collections import Counter


class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.wordRep = None

	def buildIndex(self, docs, docIDs):
		index = None
		index={}
		wordRep={}
		doc_dicts={} # dict of dicts :: docID: counter of words
		allWords=set()
		for i,docID in enumerate(docIDs):
			ar=[]
			for sentence in docs[i]: 
				ar += sentence
			doc_dicts[docID]=Counter(ar)
			allWords |= set(ar)
		
		N=len(docs)
		idf={}
		for word in allWords:
			invRep=set()
			n=0
			for docID in docIDs:
				if word in doc_dicts[docID]: 
					invRep.add(docID)
					n+=1
			
			idf[word]=math.log( (N+1)/(n+1), 10)
			index[word] = invRep

		tf_idfs={}
		doc_norms={}
		for docID,doc_dict in doc_dicts.items():
			if len(doc_dict)>0:
				tf_idf = {}
				val=0.0
				for word in allWords:
					tf_idf[word] = doc_dict[word]*idf[word] # tf*idf				
					val += tf_idf[word]*tf_idf[word]

				tf_idfs[docID]=tf_idf
				doc_norms[docID]=sqrt(val)

		wordRep['idf']=idf
		wordRep['tf_idfs']=tf_idfs
		wordRep['doc_norms']=doc_norms

		self.index = index
		self.wordRep = wordRep


	def rank(self, queries):
		doc_IDs_ordered = []
		idf = self.wordRep['idf']
		tf_idfs = self.wordRep['tf_idfs']
		doc_norms = self.wordRep['doc_norms']

		docIDs = tf_idfs.keys()
		N=len(docIDs)

		for query in queries:
			q_list=[]
			for sentence in query:
				q_list += sentence
			q_dict=Counter(q_list)

			q_norm=0.0
			for k,v in q_dict.items():

				if k in idf: q_dict[k] *= idf[k] # *idf
				else: q_dict[k] *= log( N+1, 10)
				q_norm += q_dict[k]*q_dict[k]

			q_norm=sqrt(q_norm)


			ords={}
			for docID in docIDs:
				val=0
				for word in q_dict.keys():
					if word in tf_idfs[docID]:
						val += q_dict[word]*tf_idfs[docID][word] # dot product

				if (q_norm*doc_norms[docID]): val=val/(q_norm*doc_norms[docID])
				else: val = 0
				ords[docID]=val

			doc_IDs_ordered.append(sorted(ords, key=ords.get, reverse=True))

		return doc_IDs_ordered




