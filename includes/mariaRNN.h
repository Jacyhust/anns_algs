#pragma once
#include <mutex>
#include <algorithm>
#include "utils/Preprocess.h"

class mariaV6
{
private:
	std::string index_file;
	Partition parti;
public:
	int N;
	int dim;
	//int S;
	//int L;
	//int K;
 
	std::string alg_name = "mariaV6";
public:
	mariaV6(Data& data, Partition& part_) :parti(part_) {
		N = data.N;
		dim = data.dim;
		buildIndex();
	}

	void buildIndex() {
		
	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();
	
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && (1.0f-q->resHeap.top().dist) > 
				q->norm * (parti.MaxLen[i])) break;


			//apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			auto& appr_alg = apgs[i];
			auto id = parti.EachParti[i][0];
			auto data = prep->data.val[id];
			//appr_alg->addPoint((void*)(data), (size_t)id);
			//std::mutex inlock;
			appr_alg->setEf(q->k + 100);
			auto res = appr_alg->searchKnn(q->queryPoint, q->k);
			while (!res.empty()) {
				auto top = res.top();
				res.pop();
				q->resHeap.emplace(top.second, top.first);
				while (q->resHeap.size() > q->k) q->resHeap.pop();
			}
			
		}

		while (!q->resHeap.empty()) {
			auto top = q->resHeap.top();
			q->resHeap.pop();
			q->res.emplace_back(top.id, 1.0-top.dist);
		}
		
		std::reverse(q->res.begin(), q->res.end());

		q->time_total = timer.elapsed();
	}

	//void GetTables(Preprocess& prep);
	//bool IsBuilt(const std::string& file);
	~mariaV6() {
		for (int i = 0; i < parti.numChunks; ++i) {
			delete apgs[i];
		}
		delete[] apgs;
	}
};