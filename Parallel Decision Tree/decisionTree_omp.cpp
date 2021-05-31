// Parallel implementation of Decision Tree (CART) using open-MP with OOP design
// Author: Rudransh Jaiswal

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cmath>
#include <map>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <queue>
#include <time.h>
#include <unordered_map>
using namespace std;
#define n_threads 4

class treeNode{
public:
	treeNode* leftChild;
	treeNode* rightChild;
	bool isLeaf; // whether the node is leaf or not
	double splitVal; // threshold to split
	int index; //X[ind]>=threshold => go to right child or otherwise
	int classPred; //for leaf nodes to give decision
	treeNode(){
		this->leftChild=NULL;
		this->rightChild=NULL;
		this->isLeaf=false;
		this->splitVal=0.0;
		this->index=-1;
		this->classPred=-2;
	}
};

class decisionTree{
public:
	treeNode* root;
	decisionTree()
	{
		this->root=NULL;
	}
	void fit(vector<vector<double>> X_train, vector<int> Y_train, int minNodeSize, int criteria);
	void predict(vector<vector<double>> X_test, int Y_pred[]);
	void printDecisionTree();

private:
	double split_accuracy(vector<vector<double>> X_train, vector<int> Y_train, int ind, double val);
	double H(double p);
	double split_entropy(vector<vector<double>> X_train, vector<int> Y_train, int ind, double val);
	bool is_split(vector<vector<double>> X_train, int ind, double val);
	bool isPure(vector<int> Y_train);
	int predict_class(vector<int> Y_train);
	void bestSplit(vector<vector<double>> X_train, vector<int> Y_train, int criteria, int leftIndices[], int rightIndices[], double thresh_ind[]);
	treeNode* buildDecisionTree(vector<vector<double>> X_train, vector<int> Y_train, int minNodeSize, int criteria);
	void prediction(treeNode* root, vector<vector<double>> X_test, int Y_pred[]);
};

// struct to store threshold and index
struct ti_struct {
	double splitVal;
	int index;
};

double decisionTree::split_accuracy(vector<vector<double>> X_train, vector<int> Y_train, int ind, double val){
	
	double ct=0.0;
	int n=X_train.size();
	#pragma omp parallel for reduction(+:ct) num_threads(n_threads)
	for(int i=0; i<n; i++){
		if(X_train[i][ind]>=val){
			if(Y_train[i]==1)
				ct += 1.0;
		}
		else if(Y_train[i]==-1)
			ct += 1.0;
	}
	return ct/n;
}

double decisionTree::H(double p){
	if((p==0)||(p==1))
		return 0.0;
	return -(p*log2(p) + (1-p)*log2(1-p));
}

double decisionTree::split_entropy(vector<vector<double>> X_train, vector<int> Y_train, int ind, double val){
	int l=X_train.size();
	double pr=0.0;
	double qr=0.0;
	double ql=0.0;
	#pragma omp parallel for reduction(+:pr,qr,ql) num_threads(n_threads)
	for(int i=0; i<l; i++){
		if(X_train[i][ind]>=val){
			pr++;
			if(Y_train[i]==1)
				qr++;
		}
		else if(Y_train[i]==1)
				ql++;
	}
	double pl = l-pr;
	if(pr!=0)
		qr=qr/pr;
	if(pl!=0)
		ql=ql/pl;
	pl=pl/l;
	pr=pr/l;
	return pl*H(ql)+pr*H(qr);
}

bool decisionTree::is_split(vector<vector<double>> X_train, int ind, double val){
	int l = X_train.size();
	int ct=0;
	#pragma omp parallel for reduction(+:ct) num_threads(n_threads)
	for(int i=0; i<l; i++){
		if(X_train[i][ind]>=val)
			ct++;
	}
	if((ct==0)||(ct==l))
		return false;
	return true;
}

bool decisionTree::isPure(vector<int> Y_train){
	int a = Y_train[0];
	for(int i=1; i<Y_train.size(); i++){
		if(Y_train[i]!=a)
			return false;
	} 
	return true;
}

int decisionTree::predict_class(vector<int> Y_train){
	double m= Y_train.size();

	int l=Y_train.size();
	double ct=0.0;
	#pragma omp parallel for reduction(+:ct) num_threads(n_threads)
	for(int i=0; i<l; i++){
		if(Y_train[i]==1)
			ct += 1.0;
	}	
	double ans = ct/m;
	// cout<<"::ans , ct, m ,l "<<ans<<", "<<ct<<", "<<m<<", "<<l<<endl;
	return ans >= 0.5 ? 1:-1 ;
}

void decisionTree::bestSplit(vector<vector<double>> X_train, vector<int> Y_train, int criteria, int leftIndices[], int rightIndices[], double thresh_ind[]){
	// cout<<"bestSplitStart"<<endl;
	int n = X_train.size();
	int n_feats=X_train[0].size();
	double min_curr, max_curr;

	vector<double> x_min, x_max;
	for(int j=0; j<n_feats; j++){
		min_curr=X_train[0][j];
		max_curr=X_train[0][j];

		#pragma omp parallel for reduction(max: max_curr) reduction(min: min_curr) num_threads(n_threads)
		for(int i=0; i<n; i++){
			min_curr=min(min_curr,X_train[i][j]);
			max_curr=max(max_curr,X_train[i][j]);
		}
		x_min.push_back(min_curr);
		x_max.push_back(max_curr);
	}

	int lins=15;
	double h=0.0;
	double thresh=0.0;
	double bestThresh_val;
	int bestThresh_ind;
	// accuracy criteria=1
	double best_acc=0.0;
	double curr_acc=0.0;
	double best_ent=1.0;
	double curr_ent=0.0;

	// {thresh:acc } {reduction(+:)}
	if(criteria==1){
		map<double, ti_struct> mp; 
		ti_struct t_i;

		for(int i=0; i<n_feats; i++){
			h = (x_max[i]-x_min[i])/lins;

			# pragma omp parallel for default(shared) private(t_i,curr_acc,thresh) num_threads(n_threads) reduction(max: best_acc)
			for(int j=1; j<lins; j++){
				thresh = x_min[i]+ h*j;
				curr_acc = split_accuracy(X_train,Y_train,i,thresh);
				curr_acc = max(curr_acc, 1.0 - curr_acc);
				t_i.splitVal=thresh;
				t_i.index=i;
				mp.insert({curr_acc,t_i});
				best_acc=max(best_acc, curr_acc);
			}
		}

		auto itr = mp.find(best_acc);
		t_i=itr->second;
		bestThresh_ind=t_i.index;
		bestThresh_val=t_i.splitVal;
		mp.clear();

	}

	// entropy criteria=2

	else if(criteria==2){
		unordered_map<double, ti_struct> mp; 
		ti_struct t_i;

		for(int i=0; i<n_feats; i++){
			h = (x_max[i]-x_min[i])/lins;

			#pragma omp parallel for default(shared) private(t_i,curr_ent,thresh) num_threads(n_threads) reduction(min: best_ent)
			for(int j=1; j<lins; ++j){
				thresh = x_min[i]+ h*j;
				curr_ent=split_entropy(X_train,Y_train,i,thresh);
				curr_ent=min(curr_ent,1.0-curr_ent);
				t_i.splitVal=thresh;
				t_i.index=i;
				mp.insert({curr_ent,t_i});
				best_ent=min(best_ent,curr_ent);
			}
		}

		auto itr = mp.find(best_ent);
		t_i=itr->second;
		bestThresh_ind=t_i.index;
		bestThresh_val=t_i.splitVal;
		mp.clear();

	}
	// cout<<"here1"<<endl;
	thresh_ind[0]=bestThresh_val;
	thresh_ind[1]=bestThresh_ind;
	// cout<<"here1.5"<<endl;
	int l=0,r=0;
	for(int i=0; i<n; i++){
		if(X_train[i][bestThresh_ind]>=bestThresh_val){
			rightIndices[r]=i;
			r+=1;
		}
		else{

			leftIndices[l]=i;
			l+=1;
		}
	}
	leftIndices[l]=-1;
	rightIndices[r]=-1;
	// cout<<"here2"<<endl;
	// cout<<"this is also done"<<endl;
	// cout<<"in func lI, rI: "<<leftIndices.size()<<", "<<rightIndices.size()<<endl;
	// cout<<"bestSplitEnd"<<endl;
}


treeNode* decisionTree::buildDecisionTree(vector<vector<double>> X_train, vector<int> Y_train, int minNodeSize, int criteria){
	// this has to executed sequentially
	// cout<<"bDTStart, X_train.size():"<<X_train.size()<<endl;
	// cout<<"entered"<<endl;

	// treeNode* node = new treeNode();
	// A* b = (A*)malloc(sizeof(A));
	treeNode* node = (treeNode*)malloc(sizeof(treeNode));

	int n=X_train.size();

	if(n<=minNodeSize){
		// cout<<"bDTStart_substep1a"<<endl;
		// cout<<"1a"<<endl;
		node->classPred = predict_class(Y_train);
		node->leftChild=NULL;
		node->rightChild=NULL;
		node->isLeaf=true;
		node->splitVal=-1.0;
		node->index=-1;
		// cout<<"1a~"<<endl;
		// cout<<" isLeaf: "<<node->isLeaf<<endl;

	}
	else if(isPure(Y_train)){
		// cout<<"bDTStart_substep1b"<<endl;
		// cout<<"1b"<<endl;
		node->classPred = Y_train[0];
		node->leftChild=NULL;
		node->rightChild=NULL;
		node->isLeaf=true;
		node->splitVal=-1.0;
		node->index=-1;
		// cout<<"isLeaf: "<<node->isLeaf<<endl;
		// cout<<"1b~"<<endl;
	}

	else{
		// cout<<"bDTStart_substep1c"<<endl;
		// cout<<"1c"<<endl;
		int leftIndices[n+1]={0};
		int rightIndices[n+1]={0};
		double thresh_ind[2];
		bestSplit(X_train, Y_train, criteria,leftIndices,rightIndices,thresh_ind);
		// cout<<" out func lI, rI: "<<leftIndices.size()<<", "<<rightIndices.size()<<endl;
		// cout<<"bDTStart_substep2c"<<endl;
		vector<vector<double>> X_left, X_right;
		vector<int> Y_left, Y_right;

		for(int i=0; i<n; i++){
			if(leftIndices[i]==-1)
				break;
			X_left.push_back(X_train[leftIndices[i]]);
			Y_left.push_back(Y_train[leftIndices[i]]);
		}
		for(int i=0; i<n; i++){
			if(rightIndices[i]==-1)
				break;
			X_right.push_back(X_train[rightIndices[i]]);
			Y_right.push_back(Y_train[rightIndices[i]]);
		}
		node->isLeaf=false;
		node->splitVal=thresh_ind[0];
		node->index=int(thresh_ind[1]);

		node->leftChild = buildDecisionTree(X_left,Y_left,minNodeSize,criteria);
		node->rightChild = buildDecisionTree(X_right,Y_right,minNodeSize,criteria);

		// cout<<"1c~"<<endl;

	}
	// cout<<"exited"<<endl;
	return node;

}

void decisionTree::prediction(treeNode* root, vector<vector<double>> X_test, int Y_pred[]){
	int n = X_test.size();
	treeNode* node = root;
	#pragma omp parallel for private(node) default(shared) num_threads(n_threads)
	for(int i=0; i<n; i++){
		node = root;
		while(node->isLeaf == false){
			if(X_test[i][node->index] >= node->splitVal)
				node = node->rightChild;
			else
				node=node->leftChild;
		}
		Y_pred[i]=node->classPred;
	}
}


void decisionTree::fit(vector<vector<double>> X_train, vector<int> Y_train, int minNodeSize, int criteria){
	treeNode* root = buildDecisionTree(X_train,Y_train,minNodeSize,criteria);
	this->root=root;
}

void decisionTree::predict(vector<vector<double>> X_test, int Y_pred[]){
	prediction(this->root, X_test, Y_pred);
}

void decisionTree::printDecisionTree(){
	treeNode* root=this->root;

	printf("Printing decision tree (index,splitVal,isLeaf):\n");
	queue <treeNode> bfsQ;
	int x,j;
	treeNode* nextNode;
	bfsQ.push(*root);
	cout <<"("<<root->index << ","<<root->splitVal<<","<<root->isLeaf<<")\n";
	while(bfsQ.size()!=0){
		nextNode = &(bfsQ.front());
		bfsQ.pop();
		if((nextNode->leftChild)!=NULL){
			bfsQ.push(*(nextNode->leftChild));
			cout <<"("<<nextNode->leftChild->index << ","<<nextNode->leftChild->splitVal<<","<<nextNode->leftChild->isLeaf<<") | ";
		}
		else{
			cout<<"(null) | ";
		}
		if((nextNode->rightChild)!=NULL){
			bfsQ.push(*(nextNode->rightChild));
			cout <<"("<< nextNode->rightChild->index << ","<<nextNode->rightChild->splitVal<<","<<nextNode->rightChild->isLeaf<<") | ";
		}
		else{
			cout<<"(null) | ";
		}
		cout << endl;
	}

}

vector<vector<double>> X_train;
vector<vector<double>> X_test;
vector<int> Y_train;
vector<int> Y_test;

void readCSV(string fname, string str, char delim)
{
	if(str.compare("X_train")==0){
		ifstream ifs(fname);
		string line;
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <double> values;
			while(getline(lineStream,cell,delim)){
				const char *cstr = cell.c_str();
				values.push_back(stod(cstr));
			}
			X_train.push_back(values);
		}
		ifs.close();
	}
	else if(str.compare("X_test")==0){
		ifstream ifs(fname);
		string line;
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <double> values;
			while(getline(lineStream,cell,delim)){
				const char *cstr = cell.c_str();
				values.push_back(stod(cstr));
			}
			X_test.push_back(values);
		}
		ifs.close();
	}

	else if(str.compare("Y_train")==0){
		ifstream ifs(fname);
		string line;
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			while(getline(lineStream,cell,delim)){
				const char *cstr = cell.c_str();
				Y_train.push_back(atoi(cstr));
			}
		}
		ifs.close();
	}

	else if(str.compare("Y_test")==0){
		ifstream ifs(fname);
		string line;
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			while(getline(lineStream,cell,delim)){
				const char *cstr = cell.c_str();
				Y_test.push_back(atoi(cstr));
			}
		}
		ifs.close();
	}

	else{
		cout<<"invalid input"<<endl;
	}

}

double accuracy(vector<int> Y_true, int Y_pred[]){
	int n = Y_true.size();
	double ct=0.0;
	for(int i=0; i<n; i++){
		if(Y_true[i]==Y_pred[i])
			ct += 1.0;
	}
	return (ct/n)*100;
}

int main(int argc, char** agrv){
	clock_t start, end;
	const char *criteria_arr[2] = { "accuracy", "entropy"};
	cout<<"prediction on real-world dataset : breast_cancer_data \n";
	char delim = ' ';
	readCSV("breast_cancer_data/X_train.txt", "X_train",delim);
	readCSV("breast_cancer_data/X_test.txt","X_test",delim);
	readCSV("breast_cancer_data/Y_train.txt","Y_train",delim);
	readCSV("breast_cancer_data/Y_test.txt","Y_test",delim);
	int Y_train_pred[X_train.size()];
	int Y_test_pred[X_test.size()];

	int minNodeSize = 2;
	int criteria = 2;
	start=clock_t();
	decisionTree model = decisionTree();
	model.fit(X_train,Y_train,minNodeSize,criteria);
	end=clock();

	cout<<"model built\n";
	cout<<"time : "<<(double(end - start) / double(CLOCKS_PER_SEC))<<" s "<<"| n_threads: "<<n_threads<<" | criteria: "<<criteria_arr[criteria-1]<<endl;
	model.predict(X_train,Y_train_pred);
	model.predict(X_test,Y_test_pred);
	cout<<"train acccuracy: "<<accuracy(Y_train,Y_train_pred)<<"%"<<endl;
	cout<<"test acccuracy: "<<accuracy(Y_test,Y_test_pred)<<"%"<<endl;

	// model.printDecisionTree(); //uses bfs-traversal to print the data

	return 0;
}
