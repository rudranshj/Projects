from util import *
import math
from math import log


class Evaluation():

	def __init__(self):
		self.qrels = None

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		precision = -1
		pred=set(query_doc_IDs_ordered[:k])
		truth=set(true_doc_IDs)
		precision=len(pred & truth)/k # check if denom = k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		if self.qrels is None: self.qrels = qrels

		truths={}
		for query_id in query_ids:
			truths[int(query_id)]=[]

		for qrel in qrels:
			if int(qrel["query_num"]) in truths:
				truths[int(qrel["query_num"])].append(int(qrel["id"]))

		meanPrecision = 0.0
		for i,pred in enumerate(doc_IDs_ordered):
			meanPrecision += self.queryPrecision(pred, query_ids[i], truths[query_ids[i]], k)

		meanPrecision = meanPrecision/len(query_ids)
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		recall = 0.0
		pred=set(query_doc_IDs_ordered[:k])
		truth=set(true_doc_IDs)
		recall=len(pred & truth)/len(truth) # check if denom = k

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		if self.qrels is None: self.qrels = qrels

		truths={}
		for query_id in query_ids:
			truths[int(query_id)]=[]

		for qrel in qrels:
			if int(qrel["query_num"]) in truths:
				truths[int(qrel["query_num"])].append(int(qrel["id"]))

		meanRecall = 0.0
		for i,pred in enumerate(doc_IDs_ordered):
			meanRecall += self.queryRecall(pred, query_ids[i], truths[query_ids[i]], k)

		meanRecall = meanRecall/len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall( query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if (precision + recall): fscore = (2*precision*recall)/(precision + recall)
		else: fscore=0.0
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		
		if self.qrels is None: self.qrels = qrels

		truths={}
		for query_id in query_ids:
			truths[int(query_id)]=[]

		for qrel in qrels:
			if int(qrel["query_num"]) in truths:
				truths[int(qrel["query_num"])].append(int(qrel["id"]))

		meanFscore = 0.0
		for i,pred in enumerate(doc_IDs_ordered):
			meanFscore += self.queryFscore(pred, query_ids[i], truths[query_ids[i]], k)

		meanFscore = meanFscore/len(query_ids)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		
		qrels=self.qrels
		nDCG, dcg, idcg = 0.0, 0.0, 0.0

		relevance={}
		for qrel in qrels:
			if int(qrel["query_num"]) == query_id:
				relevance[int(qrel["id"])] = 5 - int(qrel["position"])

		# ideal_rank = list(relevance.values())
		# ideal_rank.sort(reverse=True)

		kmax=min(k,len(relevance))
		obs_rank=[]
		for i,doc_id in enumerate(query_doc_IDs_ordered[:kmax]):
			if doc_id in relevance: 
				dcg += relevance[doc_id]/log(i+2, 2) # log(i+1 + 1) as i runs from 0
				obs_rank.append(relevance[doc_id])
			# idcg += ideal_rank[i]/log(i+2, 2)
		# obs_rank.sort(reverse=True)
		
		ideal_rank = list(relevance.values())
		ideal_rank.sort(reverse=True)
		for i, rel in enumerate(ideal_rank[:k]):
			idcg += rel/log(i+2, 2)

		if dcg==0.0 : 
			nDCG=0.0
		else: 
			nDCG = dcg/idcg
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
	
		if self.qrels is None: self.qrels = qrels

		meanNDCG = 0.0
		for i, query_id in enumerate(query_ids):
			meanNDCG += self.queryNDCG(doc_IDs_ordered[i], query_id, None, k)

		meanNDCG = meanNDCG/len(query_ids)
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		
		avgPrecision = 0.0
		truth=set(true_doc_IDs)

		ct=0
		for i in range(k):
			if query_doc_IDs_ordered[i] in truth:
				ct += 1
				avgPrecision += ct/(i+1)

		if ct==0: return 0
		avgPrecision = avgPrecision/ct
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		
		if self.qrels is None : self.qrels = qrels

		truths={}
		for query_id in query_ids:
			truths[int(query_id)]=[]

		for qrel in qrels:
			if int(qrel["query_num"]) in truths:
				truths[int(qrel["query_num"])].append(int(qrel["id"]))

		meanAveragePrecision = 0.0
		for i,pred in enumerate(doc_IDs_ordered):
			meanAveragePrecision += self.queryAveragePrecision(pred, query_ids[i], truths[query_ids[i]], k)


		meanAveragePrecision = meanAveragePrecision/len(query_ids)
		return meanAveragePrecision

