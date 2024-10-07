#pragma once
#include <mutex>
#include <algorithm>
#include "utils/Preprocess.h"
#include "srp.h"
#include "RNNDescent.h"
#include "rnnd.h"

//// My class for store the information of a vertex
//// Do not directly define a vectex object
//// It is only for quickly computing the address shifting and increasing the readability
// struct vertex {
//	uint64_t hashval;
//	float norm;
//	int size;
//	uint32_t* links;
// };

// struct vertex {
//	char* pt;
//	//uint64_t hashval;
//	//float norm;
//	//int size;
//	//uint32_t* links;
// };

class mariaV6
{
private:
	std::string index_file;
	Partition parti;
	std::vector<std::vector<std::vector<uint32_t>>> knngs;
	rnndescent::rnn_para para;
	std::atomic<size_t> cost{0};
	char *link_lists = nullptr;
	Data data;

public:
	int N;
	int dim;
	// int S;
	int L;
	int K;

	std::string alg_name = "mariaV6";

public:
	mariaV6(Data &data_, Partition &part_, int L_, int K_) : parti(part_)
	{
		data = data_;
		N = data.N;
		dim = data.dim;
		L = L_;
		K = K_;
		// const int min_size = 400;
		para.S = 36;
		para.T1 = 2;
		para.T2 = 4;

		lsh::timer timer;
		std::cout << "CONSTRUCTING MARIAV6..." << std::endl;
		timer.restart();
		buildIndex();
		std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl
				  << std::endl;
	}

	void buildIndex()
	{
		lsh::srp srp(data, parti.EachParti, index_file + "_srp", data.N, data.dim, L, K, 0);
		knngs.resize(parti.numChunks);
#pragma omp parallel for schedule(dynamic)
		for (int i = parti.numChunks - 1; i >= 0; --i)
		{
			if (parti.EachParti[i].size() < 400)
			{
				continue;
			}

			rnndescent::Matrix<float> base_data;
			base_data.load(parti.EachParti[i], data.base, data.dim);
			// rnndescent::MatrixOracle<float, rnndescent::metric::ip> oracle(base_data);
			rnndescent::MatrixOracle<float, rnndescent::metric::l2> oracle(base_data);
			std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, para));

			// auto start = chrono::high_resolution_clock::now();
			index->build(oracle.size(), 0);
			// auto end = chrono::high_resolution_clock::now();
			/*cout << "Elapsed time in milliseconds: "
				<< 1.0 * std::chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000
				<< " s" << endl;*/
			index->extract_index_graph(knngs[i]);
			cost += index->cost;
			// break;
		}

		cost += srp.getCost();

		std::cout << "CONSTRUCTING COST: " << (float)cost / N << std::endl;
	}

	void searchInKnng(std::vector<std::vector<uint32_t>> &apg, std::vector<int> &ids, queryN *q, int start, int ef)
	{
		auto &nngraph = apg;
		int cost = 0;
		// std::cout<<"size of knng: "<<nngraph.size()<<std::endl;
		// lsh::timer timer;
		std::priority_queue<Res> accessed_candidates;

		auto &top_candidates = q->top_candidates;

		int n = nngraph.size();
		std::vector<bool> visited(n, false);
		visited[start] = true;
		float dist = calInnerProductReverse(q->queryPoint, data[ids[start]], data.dim);
		cost++;
		accessed_candidates.emplace(start, -dist);
		top_candidates.emplace(ids[start], dist);

		while (!accessed_candidates.empty())
		{
			Res top = accessed_candidates.top();
			if (-top.dist > top_candidates.top().dist)
				break;
			accessed_candidates.pop();

			for (auto &u : nngraph[top.id])
			{
				if (visited[u])
					continue;
				visited[u] = true;
				dist = calInnerProductReverse(q->queryPoint, data[ids[u]], data.dim);
				cost++;
				accessed_candidates.emplace(u, -dist);
				top_candidates.emplace(ids[u], dist);
				if (top_candidates.size() > ef)
					top_candidates.pop();
			}
		}

		while (top_candidates.size() > q->k)
			top_candidates.pop();

		// q->resHeap
		// q->res.resize(q->k);
		// int pos = q->k;
		// while (!top_candidates.empty()) {
		//	q->res[--pos] = top_candidates.top();
		//	top_candidates.pop();
		// }
		// q->time_total = timer.elapsed();
		q->cost += cost;
	}

	void knn(queryN *q)
	{
		lsh::timer timer;
		timer.restart();

		int ef = 200;
		for (int i = parti.numChunks - 1; i >= 0; --i)
		{
			if ((!q->top_candidates.empty()) && (-(q->top_candidates.top().dist)) >
													q->norm * sqrt(parti.MaxLen[i]))
				break;

			if (parti.EachParti[i].size() < 400)
			{
				// break;
				//  std::cout<<i<<","<<parti.EachParti[i].size()<<"..."<<std::endl;
				//  exit(-1);
				auto &top_candidates = q->top_candidates;
				for (auto &x : parti.EachParti[i])
				{
					float dist = calInnerProductReverse(q->queryPoint, data[x], data.dim);

					top_candidates.emplace(x, dist);
					if (top_candidates.size() > q->k)
						top_candidates.pop();
				}
				q->cost += parti.EachParti[i].size();
				continue;
			}

			// continue;
			auto &knng = knngs[i];

			searchInKnng(knng, parti.EachParti[i], q, 0, ef);

			// break;
			////apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			// auto& appr_alg = apgs[i];
			// auto id = parti.EachParti[i][0];
			// auto data = prep->data.val[id];
			////appr_alg->addPoint((void*)(data), (size_t)id);
			////std::mutex inlock;
			// appr_alg->setEf(q->k + 100);
			// auto res = appr_alg->searchKnn(q->queryPoint, q->k);
			// while (!res.empty()) {
			//	auto top = res.top();
			//	res.pop();
			//	q->resHeap.emplace(top.second, top.first);
			//	while (q->resHeap.size() > q->k) q->resHeap.pop();
			// }
		}

		auto &top_candidates = q->top_candidates;

		q->res.resize(q->k);
		int pos = q->k;
		while (!top_candidates.empty())
		{
			q->res[--pos] = top_candidates.top();
			top_candidates.pop();
		}

		q->time_total = timer.elapsed();
	}

	// void GetTables(Preprocess& prep);
	// bool IsBuilt(const std::string& file);
	~mariaV6()
	{
		// for (int i = 0; i < parti.numChunks; ++i) {
		//	delete apgs[i];
		// }
		// delete[] apgs;
	}
};

class mariaV7
{
private:
	std::string index_file;
	Partition parti;
	std::vector<std::vector<std::vector<uint32_t>>> knngs;
	rnnd::rnn_para para;
	int *link_lists = nullptr;
	Data data;
	std::atomic<size_t> cost{0};

public:
	int N;
	int dim;
	// int S;
	int L;
	int K;

	std::string alg_name = "mariaV7";

public:
	mariaV7(Data &data_, Partition &part_, int L_, int K_) : parti(part_)
	{
		data = data_;
		N = data.N;
		dim = data.dim;
		L = L_;
		K = K_;
		// const int min_size = 400;
		para.S = 36;
		para.T1 = 2;
		para.T2 = 4;

		lsh::timer timer;
		std::cout << "CONSTRUCTING MARIAV7..." << std::endl;
		timer.restart();
		buildIndex();
		std::cout << "CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl
				  << std::endl;
	}

	void buildIndex()
	{
		lsh::srp srp(data, parti.EachParti, index_file + "_srp", data.N, data.dim, L, K, 0);

		knngs.resize(parti.numChunks);
		lsh::timer timer;
		float time = 0.0f;

#pragma omp parallel for schedule(dynamic)
		for (int i = parti.numChunks - 1; i >= 0; --i)
		{
			if (parti.EachParti[i].size() < 400)
				continue;
			Data data_in_block;
			std::vector<std::vector<Res>> knns;
			data_in_block.N = parti.EachParti[i].size();
			data_in_block.dim = data.dim;
			data_in_block.val = new float *[data_in_block.N];

			for (int j = 0; j < parti.EachParti[i].size(); ++j)
			{
				auto &id = parti.EachParti[i][j];
				data_in_block.val[j] = data[id];
			}
			timer.restart();
			srp.kjoin(knns, parti.EachParti[i], i, para.S, 20);
			time += timer.elapsed();
			rnnd::RNNDescent index(data_in_block, para);
			// index.build(data_in_block.N, 0);
			index.build(data_in_block.N, 0, knns);
			index.extract_index_graph(knngs[i]);

			cost += index.cost;
		}

		// std::cout << "CONSTRUCTING COST: " << (float)cost/N << std::endl;
		// cost+=srp.getCost();
		std::cout << "SRP SEARCH COST (s): " << time << std::endl;
	}

	void searchInKnng(std::vector<std::vector<uint32_t>> &apg, std::vector<int> &ids, queryN *q, int start, int ef)
	{
		auto &nngraph = apg;
		int cost = 0;
		// std::cout<<"size of knng: "<<nngraph.size()<<std::endl;
		// lsh::timer timer;
		std::priority_queue<Res> accessed_candidates;

		auto &top_candidates = q->top_candidates;

		int n = nngraph.size();
		std::vector<bool> visited(n, false);
		visited[start] = true;
		float dist = calInnerProductReverse(q->queryPoint, data[ids[start]], data.dim);
		cost++;
		accessed_candidates.emplace(start, -dist);
		top_candidates.emplace(ids[start], dist);

		while (!accessed_candidates.empty())
		{
			Res top = accessed_candidates.top();
			if (-top.dist > top_candidates.top().dist)
				break;
			accessed_candidates.pop();

			for (auto &u : nngraph[top.id])
			{
				if (visited[u])
					continue;
				visited[u] = true;
				dist = calInnerProductReverse(q->queryPoint, data[ids[u]], data.dim);
				cost++;
				accessed_candidates.emplace(u, -dist);
				top_candidates.emplace(ids[u], dist);
				if (top_candidates.size() > ef)
					top_candidates.pop();
			}
		}

		while (top_candidates.size() > q->k)
			top_candidates.pop();

		// q->resHeap
		// q->res.resize(q->k);
		// int pos = q->k;
		// while (!top_candidates.empty()) {
		//	q->res[--pos] = top_candidates.top();
		//	top_candidates.pop();
		// }
		// q->time_total = timer.elapsed();
		q->cost += cost;
	}

	void knn(queryN *q)
	{
		lsh::timer timer;
		timer.restart();

		int ef = 200;
		for (int i = parti.numChunks - 1; i >= 0; --i)
		{
			if ((!q->top_candidates.empty()) && (-(q->top_candidates.top().dist)) >
													q->norm * sqrt(parti.MaxLen[i]))
				break;

			if (parti.EachParti[i].size() < 400)
			{
				// std::cout<<i<<","<<parti.EachParti[i].size()<<"..."<<std::endl;
				// exit(-1);
				auto &top_candidates = q->top_candidates;
				for (auto &x : parti.EachParti[i])
				{
					float dist = calInnerProductReverse(q->queryPoint, data[x], data.dim);

					top_candidates.emplace(x, dist);
					if (top_candidates.size() > q->k)
						top_candidates.pop();
				}
				q->cost += parti.EachParti[i].size();
				// break;
				continue;
			}

			// continue;
			auto &knng = knngs[i];

			searchInKnng(knng, parti.EachParti[i], q, 0, ef);

			// break;
			////apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
			// auto& appr_alg = apgs[i];
			// auto id = parti.EachParti[i][0];
			// auto data = prep->data.val[id];
			////appr_alg->addPoint((void*)(data), (size_t)id);
			////std::mutex inlock;
			// appr_alg->setEf(q->k + 100);
			// auto res = appr_alg->searchKnn(q->queryPoint, q->k);
			// while (!res.empty()) {
			//	auto top = res.top();
			//	res.pop();
			//	q->resHeap.emplace(top.second, top.first);
			//	while (q->resHeap.size() > q->k) q->resHeap.pop();
			// }
		}

		auto &top_candidates = q->top_candidates;

		q->res.resize(q->k);
		int pos = q->k;
		while (!top_candidates.empty())
		{
			q->res[--pos] = top_candidates.top();
			top_candidates.pop();
		}

		q->time_total = timer.elapsed();
	}

	// void GetTables(Preprocess& prep);
	// bool IsBuilt(const std::string& file);
	~mariaV7()
	{
		// for (int i = 0; i < parti.numChunks; ++i) {
		//	delete apgs[i];
		// }
		// delete[] apgs;
	}
};

class mariaV8
{
private:
	std::string index_file;
	Partition &parti;
	std::vector<std::vector<std::vector<uint32_t>>> knngs; // edges in each block
	lsh::srp *srp = nullptr;
	rnnd::rnn_para para;
	int *link_lists = nullptr;
	Data data;
	std::atomic<size_t> cost{0};
	int width = 20;
	// The pairs of block that are connected to each other
	struct block_pairs
	{
		int block1_id = -1;
		int block2_id = -1;
		int S = 32;
		int efC = 32;
		std::vector<std::vector<int>> normal_edges;
		block_pairs(int i, int j, int S_, int deg_) : block1_id(i), block2_id(j), S(S_), efC(deg_) {}
	};

	std::vector<block_pairs> conn_blocks;

public:
	int N;
	int dim;
	// int S;
	int L;
	int K;
	// int max_degree = -1;
	size_t size_per_point = 16;
	float indexing_time = 0.0f;
	float *square_norms = nullptr;
	std::string alg_name = "mariaV8";
	// char* link_lists = nullptr;

public:
	mariaV8(Data &data_, float *norms, const std::string &file, Partition &part_, int L_, int K_) : parti(part_)
	{
		data = data_;
		square_norms = norms;
		N = data.N;
		dim = data.dim;
		L = L_;
		K = K_;
		index_file = file;
		// const int min_size = 400;
		para.S = 36;
		para.T1 = 2;
		para.T2 = 4;

		// para.S = 2;
		// para.T1 = 1;
		// para.T2 = 1;

		lsh::timer timer;
		if (1 || !exists_test(index_file))
		{
			float mem = (float)getCurrentRSS() / (1024 * 1024);
			buildIndex();
			float memf = (float)getCurrentRSS() / (1024 * 1024);
			indexing_time = timer.elapsed();
			std::cout << "Building time:" << indexing_time << "  seconds.\n";
			FILE *fp = nullptr;
			fopen_s(&fp, "./indexes/maria_info.txt", "a");
			if (fp)
				fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), memf - mem, indexing_time);
			saveIndex();
		}
		else
		{
			// in.close();
			srp = new lsh::srp(data, parti.EachParti, file + "_srp", data.N, data.dim);
			data = data_;
			std::cout << "Loading index from " << file << ":\n";
			float mem = (float)getCurrentRSS() / (1024 * 1024);
			loadIndex(file);
			float memf = (float)getCurrentRSS() / (1024 * 1024);
			std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
		}
	}

	void buildIndex()
	{
		// lsh::srp srp(data, parti.EachParti, data.N, data.dim, L, K);
		srp = new lsh::srp(data, parti.EachParti, index_file + "_srp", data.N, data.dim, L, K);
		// return;
		knngs.resize(parti.numChunks);
		lsh::timer timer;
		float time = 0.0f;

#pragma omp parallel for schedule(dynamic)
		for (int i = parti.numChunks - 1; i >= 0; --i)
		{
			if (parti.EachParti[i].size() < 100)
			{
				bfConstruction(i);
				continue;
			}
			// continue;
			Data data_in_block;
			std::vector<std::vector<Res>> knns;
			data_in_block.N = parti.EachParti[i].size();
			data_in_block.dim = data.dim;
			data_in_block.val = new float *[data_in_block.N];

			for (int j = 0; j < parti.EachParti[i].size(); ++j)
			{
				auto &id = parti.EachParti[i][j];
				data_in_block.val[j] = data[id];
			}
			// timer.restart();
			srp->kjoin1(knns, parti.EachParti[i], i, para.S, width);
			// time += timer.elapsed();
			rnnd::RNNDescent index(data_in_block, para);
			// index.build(data_in_block.N, 0);
			index.build(data_in_block.N, 0, knns);
			index.extract_index_graph(knngs[i]);

			cost += index.cost;
		}

		std::cout << "NN Descent    TIME: " << timer.elapsed() << "s." << std::endl
				  << std::endl;
		timer.restart();

		// std::vector<block_pairs> bps;
		auto &bps = conn_blocks;
		int SS = 32;
		// #pragma omp parallel for schedule(dynamic)
		for (int i = parti.numChunks - 1; i >= 0; --i)
		{
			int init_S = SS;
			int j = 1;
			while (i - j >= 0)
			{
				// int init_K = (2 * init_S + 32) / L;
				int init_K = (2 * init_S + SS);
				bps.emplace_back(i - j, i, init_S, init_K);
				j *= 2;
				if (init_S > 1)
					init_S /= 2;
			}
		}

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < bps.size(); ++i)
		{
			interConnection(bps[i]);
		}

		std::cout << "Inter-Connect TIME: " << timer.elapsed() << "s." << std::endl
				  << std::endl;

		// std::cout << "CONSTRUCTING COST: " << (float)cost/N << std::endl;
		// cost+=srp.getCost();
		// std::cout << "SRP SEARCH COST (s): " << time << std::endl;
	}

	void bfConstruction(int i)
	{
		int n = parti.EachParti[i].size();
		if (n <= 1)
			return;
		std::vector<std::vector<Res>> nnset(n, std::vector<Res>(n, Res(-1, FLT_MAX)));
#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < n; ++j)
		{
			for (int l = 0; l < j; ++l)
			{
				float dist = calInnerProductReverse(data[parti.EachParti[i][j]], data[parti.EachParti[i][l]], data.dim);
				nnset[j][l] = Res(l, dist);
				nnset[l][j] = Res(j, dist);
			}
		}

		// #pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < n; ++j)
		{
			std::sort(nnset[j].begin(), nnset[j].end());
		}
		auto &apg = knngs[i];
		int size = para.S;
		if (size > n)
			size = n - 1;
		apg.resize(n, std::vector<uint32_t>(size));
		for (int j = 0; j < n; ++j)
		{
			for (int l = 0; l < size; ++l)
			{
				apg[j][l] = nnset[j][l].id;
			}
		}
	}

	// np1<np2
	void interConnection(block_pairs &bp)
	{
		std::vector<std::vector<Res>> knns;
		srp->kjoin(knns, parti.EachParti[bp.block1_id], bp.block1_id,
				   parti.EachParti[bp.block2_id], bp.block2_id, para.S, width);

		auto &knng1 = knngs[bp.block1_id];
		bp.normal_edges.resize(parti.EachParti[bp.block2_id].size());
		// std::vector<int> visited(knng2.size(), -1);
#pragma omp parallel for schedule(dynamic, 256)
		for (int i = 0; i < knns.size(); ++i)
		{
			float *q = data[parti.EachParti[bp.block2_id][i]];
			std::priority_queue<Res> top_candidates, candidate_set;
			std::vector<bool> visited(knng1.size(), false);
			for (auto &res : knns[i])
			{
				top_candidates.push(res);
				res.dist *= -1.0f;
				candidate_set.push(res);
				visited[res.id] = true;
			}

			while (top_candidates.size() > bp.efC)
				top_candidates.pop();

			while (!candidate_set.empty())
			{
				auto top = candidate_set.top();
				candidate_set.pop();
				if (-top.dist > top_candidates.top().dist)
					break;
				for (auto &u : knng1[top.id])
				{
					if (visited[u])
						continue;
					visited[u] = true;
					float dist = cal_inner_product(q, data[parti.EachParti[bp.block1_id][u]], dim);
					candidate_set.emplace(u, dist);
					top_candidates.emplace(u, -dist);
					if (top_candidates.size() > bp.efC)
						top_candidates.pop();
				}
			}

			while (top_candidates.size() > bp.S)
				top_candidates.pop();

			bp.normal_edges[i].reserve(top_candidates.size());
			for (int j = 0; j < top_candidates.size(); ++j)
			{
				auto &top = top_candidates.top();
				bp.normal_edges[i].push_back(top.id);
				top_candidates.pop();
			}
		}
	}

	void searchInKnng(std::vector<std::vector<uint32_t>> &apg, std::vector<int> &ids, queryN *q, int start, int ef)
	{
		auto &nngraph = apg;
		int cost = 0;
		std::priority_queue<Res> accessed_candidates;
		auto &top_candidates = q->top_candidates;
		int n = nngraph.size();
		std::vector<bool> visited(n, false);
		visited[start] = true;
		float dist = calInnerProductReverse(q->queryPoint, data[ids[start]], data.dim);
		cost++;
		accessed_candidates.emplace(start, -dist);
		top_candidates.emplace(ids[start], dist);

		while (!accessed_candidates.empty())
		{
			Res top = accessed_candidates.top();
			if (-top.dist > top_candidates.top().dist)
				break;
			accessed_candidates.pop();

			for (auto &u : nngraph[top.id])
			{
				if (visited[u])
					continue;
				visited[u] = true;
				dist = calInnerProductReverse(q->queryPoint, data[ids[u]], data.dim);
				cost++;
				accessed_candidates.emplace(u, -dist);
				top_candidates.emplace(ids[u], dist);
				if (top_candidates.size() > ef)
					top_candidates.pop();
			}
		}

		while (top_candidates.size() > q->k)
			top_candidates.pop();
		q->cost += cost;
	}

	void knn(queryN *q)
	{
		lsh::timer timer;
		timer.restart();

		int ef = 180;
		for (int i = parti.numChunks - 1; i >= 0; --i)
		{
			if ((!q->top_candidates.empty()) && (-(q->top_candidates.top().dist)) >
													q->norm * sqrt(parti.MaxLen[i]))
				break;

			if (parti.EachParti[i].size() < 400)
			{
				// std::cout<<i<<","<<parti.EachParti[i].size()<<"..."<<std::endl;
				// exit(-1);
				auto &top_candidates = q->top_candidates;
				for (auto &x : parti.EachParti[i])
				{
					float dist = calInnerProductReverse(q->queryPoint, data[x], data.dim);

					top_candidates.emplace(x, dist);
					if (top_candidates.size() > q->k)
						top_candidates.pop();
				}
				q->cost += parti.EachParti[i].size();
				// break;
				continue;
			}

			// continue;
			auto &knng = knngs[i];

			searchInKnng(knng, parti.EachParti[i], q, 0, ef);

			// break;
		}

		auto &top_candidates = q->top_candidates;

		q->res.resize(q->k);
		int pos = q->k;
		while (!top_candidates.empty())
		{
			q->res[--pos] = top_candidates.top();
			top_candidates.pop();
		}

		q->time_total = timer.elapsed();
	}

	// void GetTables(Preprocess& prep);
	// bool IsBuilt(const std::string& file);

	void compute_maxsize()
	{
		int i = parti.numChunks - 1;
		int init_S = 32;
		int j = 1;
		size_per_point += para.S;
		while (i - j >= 0)
		{
			size_per_point += init_S;
			j *= 2;
			if (init_S > 1)
				init_S /= 2;
		}

		// To align with 64B
		size_per_point = ((size_per_point - 1) / 16 + 1) * 16;
	}

	void saveIndex()
	{
		compute_maxsize();
		std::cout << "Saving edges: " << std::endl;
		// delete the original vectors for saving memory
		if (N > 1e8)
			delete[] data.base;
		lsh::timer timer;
		link_lists = new int[size_per_point * N];
		for (int i = 0; i < N; ++i)
		{
			int *v = link_lists + size_per_point * i;
			// memcpy((void*)(v), (void*)(srp->hashvals[i].data()), sizeof(uint64_t));//v->hashval has the same address with v
			// memcpy(reinterpret_cast<void *>(v), reinterpret_cast<void *>(srp->hashvals[i].data()), sizeof(uint64_t));
			// float *pvnorm = (float *)(v + 2);
			// *pvnorm = sqrt(square_norms[i]);
			// int *pvsize = (int *)(v + 3);
			// *pvsize = 0;
			v[3]=0;
		}

		std::cout << "INITIA  TIME: " << timer.elapsed() << "s." << std::endl;
		timer.restart();

		for (int i = 0; i < parti.numChunks; ++i)
		{
			auto &knng = knngs[i];
			auto &ids = parti.EachParti[i];
//#pragma omp parallel for schedule(dynamic, 256)
			for (int j = 0; j < knng.size(); ++j)
			{
				int id1 = ids[j];
				int *v = (link_lists + size_per_point * id1);
				int *links = (v + 4);
				for (auto &u : knng[j])
				{
					
					//int *size = ((int *)(v + 3));
					links[(v[3])] = ids[u];
					v[3]=v[3]+1;
				}
			}
		}

		std::cout << "TANGEN  TIME: " << timer.elapsed() << "s." << std::endl;
		timer.restart();

// 		for (auto &bps : conn_blocks)
// 		{
// 			auto &ids1 = parti.EachParti[bps.block1_id];
// 			auto &ids2 = parti.EachParti[bps.block2_id];
// 			auto &knng = bps.normal_edges;
// #pragma omp parallel for schedule(dynamic, 256)
// 			for (int j = 0; j < knng.size(); ++j)
// 			{
// 				int id1 = ids2[j];
// 				// vertex* v = (vertex*)(link_lists + size_per_point * id1);
// 				int *v = (link_lists + size_per_point * id1);
// 				int *links = (v + 4);
// 				int *size = ((int *)(v + 3));
// 				for (auto &u : knng[j])
// 				{
// 					// v->links[v->size++] = ids2[u];

// 					links[(*size)++] = ids1[u];
// 					// auto& size = *((int*)(v + 12));
// 					// links[size++] = ids2[u];
// 				}
// 			}
// 		}

// 		std::cout << "NORMAL TIME: " << timer.elapsed() << "s." << std::endl;
// 		timer.restart();

		std::string file = index_file;
		std::ofstream out(file, std::ios::binary);
		out.write((char *)(&N), sizeof(int));
		out.write((char *)(&size_per_point), sizeof(size_t));
		out.write((char *)(link_lists), sizeof(int) * size_per_point * N);

		float mem = size_per_point * N * 4;
		mem /= (1 << 30);
		std::cout << "size per p : " << size_per_point << std::endl;
		std::cout << "File size  : " << mem << "GB." << std::endl;
		std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl;
	}

	void loadIndex(const std::string &file)
	{
		std::ifstream in(file, std::ios::binary);
		if (!in.good())
		{
			std::cerr << "Cannot open file:" << file << std::endl;
		}
		in.read((char *)(&N), sizeof(int));
		in.read((char *)(&size_per_point), sizeof(size_t));
		link_lists = new int[size_per_point * N];
		in.read((char *)(link_lists), sizeof(int) * size_per_point * N);
	}

	void showInfo()
	{
		std::cout << "This is the info of V8:" << std::endl;
		auto knng = knngs[0];
		auto ids = parti.EachParti[0];
		for (int i = 0; i < 10; ++i)
		{
			int j = 0;
			for (j = 0; i < N; ++j)
			{
				if (ids[j] == i)
					break;
			}

			printf("point-%d has %d neighbors:\n",i,knng[j].size());

			for(auto& x:knng[j]){
				printf("%d\t",ids[x]);
			}
			printf("\n");
		}
	}

	~mariaV8()
	{
		// for (int i = 0; i < parti.numChunks; ++i) {
		//	delete apgs[i];
		// }
		// delete[] apgs;
	}
};

class LiteMARIA
{
public:
	std::string alg_name = "LiteMaria";
	int *link_lists = nullptr;
	size_t size_per_point = 16;
	lsh::srp *srp = nullptr;
	Data data;
	int N = 0;

public:
	int ef = 200;
	// Only allow to initialize this class by reading the file
	LiteMARIA(Data &data_, const std::string &file, Partition &parti)
	{
		data = data_;
		srp = new lsh::srp(data, parti.EachParti, file + "_srp", data.N, data.dim);
		std::cout << "Loading index from " << file << ":\n";
		float mem = (float)getCurrentRSS() / (1024 * 1024);
		loadIndex(file);
		float memf = (float)getCurrentRSS() / (1024 * 1024);
		std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
	}

	void loadIndex(const std::string &file)
	{
		std::ifstream in(file, std::ios::binary);
		if (!in.good())
		{
			std::cerr << "Cannot open file:" << file << std::endl;
		}
		in.read((char *)(&N), sizeof(int));
		in.read((char *)(&size_per_point), sizeof(size_t));
		link_lists = new int[size_per_point * N];
		in.read((char *)(link_lists), sizeof(int) * size_per_point * N);
	}

	void showInfo()
	{
		std::cout << "This is the info of Lite:" << std::endl;
		//auto knng = knngs[0];
		//auto ids = parti.EachParti[0];
		for (int i = 0; i < 10; ++i)
		{
			int *v = (link_lists + size_per_point * i);

			printf("point-%d has %d neighbors:\n",i,v[3]);

			for(int j=0;j<v[3];++j){
				printf("%d\t",v[j+4]);
			}
			printf("\n");
		}
	}

	void knn(queryN *q)
	{
		std::priority_queue<Res> top_candidates, candidate_set;
		srp->knn(q);
		std::vector<bool> &visited = q->visited;
		int efS = q->k + ef;
		while (!(q->top_candidates.empty()))
		{
			auto top = q->top_candidates.top();
			candidate_set.emplace(top.id, -top.dist);
			top_candidates.emplace(top.id,top.dist);
			q->top_candidates.pop();
		}

		while (top_candidates.size() > efS)
			top_candidates.pop();

		while (!candidate_set.empty())
		{
			auto top = candidate_set.top();
			candidate_set.pop();
			if (-top.dist > top_candidates.top().dist)
				break;
			int *v = (link_lists + size_per_point * top.id);
			// int size = *((int*)(v + 3));
			int size = v[3];
			int *links = (v + 4);
			for (int i = 0; i < size; ++i)
			{
				auto &u = links[i];
				if (visited[u])
					continue;
				visited[u] = true;
				float dist = cal_inner_product(q->queryPoint, data[u], data.dim);
				candidate_set.emplace(u, dist);
				top_candidates.emplace(u, -dist);
				if (top_candidates.size() > efS)
					top_candidates.pop();
			}
		}

		while (top_candidates.size() > q->k)
			top_candidates.pop();

		q->res.clear();
		q->res.reserve(top_candidates.size());
		while (!top_candidates.empty())
		{
			auto top = top_candidates.top();
			q->res.emplace_back(top.id, -top.dist);
			top_candidates.pop();
		}
		//std::reverse(q->res.begin(), q->res.end());
		//std::vector<bool>().swap(q->visited);
	}
};