from util import *


import math
import numpy as np
import pandas as pd
from math import log
from math import sqrt
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Using LSA
class InformationRetrieval():

	def __init__(self):
		self.tfidf_combined_df=None
		self.n_docs=0
		self.n_comps=500 #for svd
		self.vec_rep_docs_trunc=None
		self.vec_rep_qry_trunc=None
		

	def buildIndex(self, docs, docIDs, queries):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		arg3 : 
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		Returns
		-------
		None
		"""

		doc_dicts={} # dict of dicts :: docID: counter of words
		allWords=set()
		tf_idfs_doc={}
		idf={}

		for i,docID in enumerate(docIDs):
			ar=[]
			for sentence in docs[i]: 
				ar += sentence
			doc_dicts[docID]=Counter(ar)
			allWords |= set(ar)
		
		N=len(docs)
		for word in allWords:
			n=0
			for docID in docIDs:
				if word in doc_dicts[docID]: n+=1
			idf[word]=round(math.log( (N+1)/(n+1), 10),5)


		
		for docID, doc_dict in doc_dicts.items():
			tf_idf={}
			if len(doc_dict)==0:
				tf_idfs_doc[docID]=tf_idf
				continue

			for word in doc_dict.keys():
				tf_idf[word] = doc_dict[word]*idf[word] # tf*idf

			tf_idfs_doc[docID]=tf_idf

		

		tf_idfs_qry={}
		for i,query in enumerate(queries):
			q_list=[]
			for sentence in query:
				q_list += sentence

			tf_idf=Counter(q_list)

			for word in tf_idf.keys():
				if word in idf: tf_idf[word] *= idf[word]
				else: tf_idf[word] *= log(N+1, 10)

			tf_idfs_qry[N+i+1]=tf_idf

		

		tfidf_combined=tf_idfs_doc.copy()
		tfidf_combined.update(tf_idfs_qry)
		tfidf_combined_df=pd.DataFrame.from_dict(tfidf_combined)
		tfidf_combined_df=tfidf_combined_df.fillna(0)
		self.tfidf_combined_df=tfidf_combined_df
		self.n_docs=N

		vec_rep=tfidf_combined_df.values
		vec_rep_docs = np.array(vec_rep[:,0:N])
		vec_rep_qry = np.array(vec_rep[:,N:])

		svd=TruncatedSVD(n_components=self.n_comps,random_state=16108)
		svd.fit(vec_rep_docs.T)
		vec_rep_docs_trunc=svd.transform(vec_rep_docs.T).T
		vec_rep_qry_trunc=svd.transform(vec_rep_qry.T).T

		self.vec_rep_docs_trunc=vec_rep_docs_trunc
		self.vec_rep_qry_trunc=vec_rep_qry_trunc
		# vec_rep_docs_trunc=normalize(vec_rep_docs_trunc, norm='l2', axis=0)
		# vec_rep_qry_trunc=normalize(vec_rep_qry_trunc, norm='l2', axis=0)



	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		vec_rep_docs_trunc=self.vec_rep_docs_trunc
		vec_rep_qry_trunc=self.vec_rep_qry_trunc
		vec_rep_docs_trunc=normalize(vec_rep_docs_trunc, norm='l2', axis=0)
		vec_rep_qry_trunc=normalize(vec_rep_qry_trunc, norm='l2', axis=0)
		cos_sim_matrix = np.matmul(vec_rep_qry_trunc.T, vec_rep_docs_trunc)
		docs_ID_retreived=[]
		for i in range(len(queries)):
		    docs_ID_retreived.append(1+np.argsort(cos_sim_matrix[i])[::-1])
		return docs_ID_retreived





# from util import *


# import math
# from math import log
# from math import sqrt
# from collections import Counter

# # Using LSA
# class InformationRetrieval():

# 	def __init__(self):
# 		self.index = None
# 		self.wordRep = None


# 	def buildIndex(self, docs, docIDs):
# 		"""
# 		Builds the document index in terms of the document
# 		IDs and stores it in the 'index' class variable

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of lists where each sub-list is
# 			a document and each sub-sub-list is a sentence of the document
# 		arg2 : list
# 			A list of integers denoting IDs of the documents
# 		Returns
# 		-------
# 		None
# 		"""

# 		index={}
# 		wordRep={}
# 		doc_dicts={} # dict of dicts :: docID: counter of words
# 		allWords=set()
# 		for i,docID in enumerate(docIDs):
# 			ar=[]
# 			for sentence in docs[i]: 
# 				ar += sentence
# 			doc_dicts[docID]=Counter(ar)
# 			allWords |= set(ar)
	
# 		N=len(docs)
# 		idf={}
# 		for word in allWords:
# 			invRep=set()
# 			n=0
# 			for docID in docIDs:
# 				if word in doc_dicts[docID]: 
# 					invRep.add(docID)
# 					n+=1
			
# 			idf[word]=round(math.log( (N+1)/(n+1), 10),5)
# 			index[word] = invRep

# 		tf_idfs={}
# 		doc_norms={}
# 		for docID,doc_dict in doc_dicts.items():
# 			if len(doc_dict)>0:
# 				tf_idf = {}
# 				val=0.0
# 				for word in allWords:
# 					tf_idf[word] = doc_dict[word]*idf[word] # tf*idf				
# 					val += tf_idf[word]*tf_idf[word]

# 				tf_idfs[docID]=tf_idf
# 				doc_norms[docID]=sqrt(val)

# 		wordRep['idf']=idf
# 		wordRep['tf_idfs']=tf_idfs
# 		wordRep['doc_norms']=doc_norms

# 		self.index = index
# 		self.wordRep = wordRep


# 	def rank(self, queries):
# 		"""
# 		Rank the documents according to relevance for each query

# 		Parameters
# 		----------
# 		arg1 : list
# 			A list of lists of lists where each sub-list is a query and
# 			each sub-sub-list is a sentence of the query
		

# 		Returns
# 		-------
# 		list
# 			A list of lists of integers where the ith sub-list is a list of IDs
# 			of documents in their predicted order of relevance to the ith query
# 		"""

# 		doc_IDs_ordered = []

# 		#Fill in code here

# 		idf = self.wordRep['idf']
# 		tf_idfs = self.wordRep['tf_idfs']
# 		doc_norms = self.wordRep['doc_norms']

# 		docIDs = tf_idfs.keys()
# 		N=len(docIDs)

# 		for query in queries:
# 			q_list=[]
# 			for sentence in query:
# 				q_list += sentence
# 			q_dict=Counter(q_list)

# 			q_norm=0.0
# 			for k,v in q_dict.items():

# 				if k in idf: q_dict[k] *= idf[k] # *idf
# 				else: q_dict[k] *= log( N+1, 10)
# 				q_norm += q_dict[k]*q_dict[k]

# 			q_norm=sqrt(q_norm)


# 			ords={}
# 			for docID in docIDs:
# 				val=0
# 				for word in q_dict.keys():
# 					if word in tf_idfs[docID]:
# 						val += q_dict[word]*tf_idfs[docID][word] # dot product

# 				if (q_norm*doc_norms[docID]): val=val/(q_norm*doc_norms[docID])
# 				else: val = 0
# 				ords[docID]=val

# 			doc_IDs_ordered.append(sorted(ords, key=ords.get, reverse=True))

# 		return doc_IDs_ordered




