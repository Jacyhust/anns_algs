#pragma once
#include <mutex>
#include <algorithm>
#include "utils/Preprocess.h"
#include "srp.h"
#include "RNNDescent.h"
#include "rnnd.h"

class mariaV6
{
private:
	std::string index_file;
	Partition parti;
	std::vector<std::vector<std::vector<uint32_t>>> knngs;
	rnndescent::rnn_para para;
	int* link_lists = nullptr;
	Data data;
public:
	int N;
	int dim;
	//int S;
	int L;
	int K;
 
	std::string alg_name = "mariaV6";
public:
	mariaV6(Data& data_, Partition& part_, int L_, int K_) :parti(part_) {
		data = data_;
		N = data.N;
		dim = data.dim;
		L = L_;
		K = K_;
		//const int min_size = 400;
		para.S = 6;
		para.T1 = 5;
		para.T2 = 8;

		lsh::timer timer;
		std::cout << "CONSTRUCTING MARIAV6..." << std::endl;
		timer.restart();
		buildIndex();
		std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
		
	}

	void buildIndex() {
		lsh::srp srp(data, parti.EachParti, data.N, data.dim, L, K);

		knngs.resize(parti.numChunks);

// #pragma omp parallel for schedule(dynamic)
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			//if (parti.EachParti[i].size() < para.S) {
			//	int num = parti.EachParti[i].size();
			//	auto& knng = knngs[i];
			//	knng.resize(num);
			//	for (int l = 0; l < num; ++l) {
			//		knng[l].reserve(num - 1);
			//		for (int j = 0; j < num; ++j) {
			//			if (l != j) knng[l].emplace_back(j);
			//		}
			//	}

			//	continue;
			//}

			if (parti.EachParti[i].size() < 400) {
				continue;
			}

			rnndescent::Matrix<float> base_data;
			base_data.load(parti.EachParti[i], data.base, data.dim);
			//rnndescent::MatrixOracle<float, rnndescent::metric::ip> oracle(base_data);
			rnndescent::MatrixOracle<float, rnndescent::metric::l2> oracle(base_data);
			std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, para));
			
			//auto start = chrono::high_resolution_clock::now();
			index->build(oracle.size(), 0);
			//auto end = chrono::high_resolution_clock::now();
			/*cout << "Elapsed time in milliseconds: "
				<< 1.0 * std::chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000
				<< " s" << endl;*/
			index->extract_index_graph(knngs[i]);

			//break;
		}


	}

	void searchInKnng(std::vector<std::vector<uint32_t>>& apg, std::vector<int>& ids, queryN* q, int start, int ef) {
		auto& nngraph = apg;
		int cost = 0;
		//std::cout<<"size of knng: "<<nngraph.size()<<std::endl;
		//lsh::timer timer;
		std::priority_queue<Res> accessed_candidates;

		auto& top_candidates = q->resHeap;

		int n = nngraph.size();
		std::vector<bool> visited(n, false);
		visited[start] = true;
		float dist = calInnerProductReverse(q->queryPoint, data[ids[start]], data.dim);
		cost++;
		accessed_candidates.emplace(start, -dist);
		top_candidates.emplace(ids[start], dist);

		while (!accessed_candidates.empty()) {
			Res top = accessed_candidates.top();
			if (-top.dist > top_candidates.top().dist) break;
			accessed_candidates.pop();

			for (auto& u : nngraph[top.id]) {
				if (visited[u]) continue;
				visited[u] = true;
				dist = calInnerProductReverse(q->queryPoint, data[ids[u]], data.dim);
				cost++;
				accessed_candidates.emplace(u, -dist);
				top_candidates.emplace(ids[u], dist);
				if (top_candidates.size() > ef) top_candidates.pop();
			}
		}

		while (top_candidates.size() > q->k) top_candidates.pop();

		//q->resHeap
		//q->res.resize(q->k);
		//int pos = q->k;
		//while (!top_candidates.empty()) {
		//	q->res[--pos] = top_candidates.top();
		//	top_candidates.pop();
		//}
		//q->time_total = timer.elapsed();
		q->cost += cost;
	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();
		
		int ef = 200;
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && (-(q->resHeap.top().dist)) > 
				q->norm * sqrt(parti.MaxLen[i])) break;

			if (parti.EachParti[i].size() < 400) {
				//break;
				// std::cout<<i<<","<<parti.EachParti[i].size()<<"..."<<std::endl;
				// exit(-1);
				auto& top_candidates = q->resHeap;
				for (auto& x : parti.EachParti[i]) {
					float dist = calInnerProductReverse(q->queryPoint, data[x], data.dim);

					top_candidates.emplace(x, dist);
					if (top_candidates.size() > q->k) top_candidates.pop();
				}
				q->cost+=parti.EachParti[i].size();
				continue;
			}

			//continue;
			auto& knng = knngs[i];
			
			searchInKnng(knng, parti.EachParti[i], q, 0, ef);

			//break;
			////apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			//auto& appr_alg = apgs[i];
			//auto id = parti.EachParti[i][0];
			//auto data = prep->data.val[id];
			////appr_alg->addPoint((void*)(data), (size_t)id);
			////std::mutex inlock;
			//appr_alg->setEf(q->k + 100);
			//auto res = appr_alg->searchKnn(q->queryPoint, q->k);
			//while (!res.empty()) {
			//	auto top = res.top();
			//	res.pop();
			//	q->resHeap.emplace(top.second, top.first);
			//	while (q->resHeap.size() > q->k) q->resHeap.pop();
			//}
			
		}

		auto& top_candidates = q->resHeap;
		
		q->res.resize(q->k);
		int pos = q->k;
		while (!top_candidates.empty()) {
			q->res[--pos] = top_candidates.top();
			top_candidates.pop();
		}

		q->time_total = timer.elapsed();
	}



	//void GetTables(Preprocess& prep);
	//bool IsBuilt(const std::string& file);
	~mariaV6() {
		//for (int i = 0; i < parti.numChunks; ++i) {
		//	delete apgs[i];
		//}
		//delete[] apgs;
	}
};

class mariaV7
{
private:
	std::string index_file;
	Partition parti;
	std::vector<std::vector<std::vector<uint32_t>>> knngs;
	rnnd::rnn_para para;
	int* link_lists = nullptr;
	Data data;
public:
	int N;
	int dim;
	//int S;
	int L;
	int K;

	std::string alg_name = "mariaV7";
public:
	mariaV7(Data& data_, Partition& part_, int L_, int K_) :parti(part_) {
		data = data_;
		N = data.N;
		dim = data.dim;
		L = L_;
		K = K_;
		//const int min_size = 400;
		para.S = 36;
		para.T1 = 2;
		para.T2 = 4;

		lsh::timer timer;
		std::cout << "CONSTRUCTING MARIAV7..." << std::endl;
		timer.restart();
		buildIndex();
		std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	}

	void buildIndex() {
		lsh::srp srp(data, parti.EachParti, data.N, data.dim, L, K);

		knngs.resize(parti.numChunks);

		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if (parti.EachParti[i].size() < 400) continue;
			Data data_in_block;
			std::vector<std::vector<Res>> knns;
			data_in_block.N = parti.EachParti[i].size();
			data_in_block.dim = data.dim;
			data_in_block.val = new float* [data_in_block.N];

			for (int j = 0; j < parti.EachParti[i].size(); ++j) {
				auto& id = parti.EachParti[i][j];
				data_in_block.val[j] = data[id];
			}
			srp.kjoin(knns, parti.EachParti[i], i, para.S, 20);
			rnnd::RNNDescent index(data_in_block, para);
			//index.build(data_in_block.N, 0);
			index.build(data_in_block.N, 0, knns);
			index.extract_index_graph(knngs[i]);
		}


	}

	void searchInKnng(std::vector<std::vector<uint32_t>>& apg, std::vector<int>& ids, queryN* q, int start, int ef) {
		auto& nngraph = apg;
		int cost = 0;
		//std::cout<<"size of knng: "<<nngraph.size()<<std::endl;
		//lsh::timer timer;
		std::priority_queue<Res> accessed_candidates;

		auto& top_candidates = q->resHeap;

		int n = nngraph.size();
		std::vector<bool> visited(n, false);
		visited[start] = true;
		float dist = calInnerProductReverse(q->queryPoint, data[ids[start]], data.dim);
		cost++;
		accessed_candidates.emplace(start, -dist);
		top_candidates.emplace(ids[start], dist);

		while (!accessed_candidates.empty()) {
			Res top = accessed_candidates.top();
			if (-top.dist > top_candidates.top().dist) break;
			accessed_candidates.pop();

			for (auto& u : nngraph[top.id]) {
				if (visited[u]) continue;
				visited[u] = true;
				dist = calInnerProductReverse(q->queryPoint, data[ids[u]], data.dim);
				cost++;
				accessed_candidates.emplace(u, -dist);
				top_candidates.emplace(ids[u], dist);
				if (top_candidates.size() > ef) top_candidates.pop();
			}
		}

		while (top_candidates.size() > q->k) top_candidates.pop();

		//q->resHeap
		//q->res.resize(q->k);
		//int pos = q->k;
		//while (!top_candidates.empty()) {
		//	q->res[--pos] = top_candidates.top();
		//	top_candidates.pop();
		//}
		//q->time_total = timer.elapsed();
		q->cost += cost;
	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();

		int ef = 200;
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && (-(q->resHeap.top().dist)) >
				q->norm * sqrt(parti.MaxLen[i])) break;

			if (parti.EachParti[i].size() < 400) {
				// std::cout<<i<<","<<parti.EachParti[i].size()<<"..."<<std::endl;
				// exit(-1);
				auto& top_candidates = q->resHeap;
				for (auto& x : parti.EachParti[i]) {
					float dist = calInnerProductReverse(q->queryPoint, data[x], data.dim);

					top_candidates.emplace(x, dist);
					if (top_candidates.size() > q->k) top_candidates.pop();
				}
				q->cost += parti.EachParti[i].size();
				continue;
			}

			//continue;
			auto& knng = knngs[i];

			searchInKnng(knng, parti.EachParti[i], q, 0, ef);

			//break;
			////apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			//auto& appr_alg = apgs[i];
			//auto id = parti.EachParti[i][0];
			//auto data = prep->data.val[id];
			////appr_alg->addPoint((void*)(data), (size_t)id);
			////std::mutex inlock;
			//appr_alg->setEf(q->k + 100);
			//auto res = appr_alg->searchKnn(q->queryPoint, q->k);
			//while (!res.empty()) {
			//	auto top = res.top();
			//	res.pop();
			//	q->resHeap.emplace(top.second, top.first);
			//	while (q->resHeap.size() > q->k) q->resHeap.pop();
			//}

		}

		auto& top_candidates = q->resHeap;

		q->res.resize(q->k);
		int pos = q->k;
		while (!top_candidates.empty()) {
			q->res[--pos] = top_candidates.top();
			top_candidates.pop();
		}

		q->time_total = timer.elapsed();
	}



	//void GetTables(Preprocess& prep);
	//bool IsBuilt(const std::string& file);
	~mariaV7() {
		//for (int i = 0; i < parti.numChunks; ++i) {
		//	delete apgs[i];
		//}
		//delete[] apgs;
	}
};