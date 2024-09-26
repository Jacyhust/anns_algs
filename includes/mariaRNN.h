#pragma once
#include <mutex>
#include <algorithm>
#include "utils/Preprocess.h"
#include "srp.h"
#include "RNNDescent.h"

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
		para.S = 36;
		para.T1 = 2;
		para.T2 = 8;
		buildIndex();
	}

	void buildIndex() {
		lsh::srp srp(data, parti.EachParti, data.N, data.dim, L, K);

		knngs.resize(parti.numChunks);

		for (int i = parti.numChunks - 1; i >= 0; --i) {
			rnndescent::Matrix<float> base_data;

			base_data.load(parti.EachParti[i], data.base, data.dim);
			rnndescent::MatrixOracle<float, rnndescent::metric::ip> oracle(base_data);
			std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, para));
			//auto start = chrono::high_resolution_clock::now();
			index->build(oracle.size(), true);
			//auto end = chrono::high_resolution_clock::now();
			/*cout << "Elapsed time in milliseconds: "
				<< 1.0 * std::chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000
				<< " s" << endl;*/
			index->extract_index_graph(knngs[i]);
		}


	}

	void knn(queryN* q) {
		lsh::timer timer;
		timer.restart();
	
		for (int i = parti.numChunks - 1; i >= 0; --i) {
			if ((!q->resHeap.empty()) && (1.0f-q->resHeap.top().dist) > 
				q->norm * (parti.MaxLen[i])) break;


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
		//for (int i = 0; i < parti.numChunks; ++i) {
		//	delete apgs[i];
		//}
		//delete[] apgs;
	}
};