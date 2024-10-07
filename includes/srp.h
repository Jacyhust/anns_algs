#pragma once
#include "utils/StructType.h"
#include "utils/Preprocess.h"
#include <cmath>
#include <assert.h>
#include <vector>
#include <queue>
#include <cfloat>

// #define USE_BLAS

// #ifdef USE_BLAS
#if defined(__GNUC__) && defined(USE_BLAS)
#include <cblas.h>
#endif

namespace lsh
{
	struct srpPair {
		uint32_t val = 0;
		int id = -1;

		srpPair() = default;
		srpPair(int id_, uint32_t hashval) : id(id_), val(hashval) {}

		bool operator<(const srpPair& rhs) const { return val < rhs.val; }
	};



	// My implement for a simple sign random prejection LSH function class
	class srp
	{
		// int N=0;

		//N * L;

		std::vector<std::vector<srpPair>> hash_tables;
		std::vector<std::vector<int>>& part_map;
		Data data;
		std::string index_file;
		std::atomic<size_t> cost{ 0 };
		float* rndAs = nullptr;
		int dim = 0;
		// Number of hash functions
		int S = 0;
		// #L Tables;
		int L = 0;
		// Dimension of the hash table
		int K = 0;
		float indexing_time = 0.0f;
		public:
		std::vector<std::vector<uint16_t>> hashvals;
		size_t getCost()
		{
			return cost;
		}

		srp() = default;

		srp(Data& data_, std::vector<std::vector<int>>& part_map_, const std::string& index_file_,
			int N_, int dim_, int L_ = 4, int K_ = 16, bool isbuilt = 1) :part_map(part_map_)
		{
			data = data_;
			// N=N_;
			dim = dim_;
			L = L_;
			K = K_;
			S = L * K;
			hashvals.resize(N_);
			index_file = index_file_;
			if (L > 4 || K > 16) {
				std::cerr << "The valid ranges of L and K are: 1<=L<=4, 1<=K<=16" << std::endl;
				exit(-1);
			}

			//std::ifstream in(index_file, std::ios::binary);
			lsh::timer timer;
			if (!(isbuilt && exists_test(index_file))) {
				float mem = (float)getCurrentRSS() / (1024 * 1024);
				buildIndex();
				float memf = (float)getCurrentRSS() / (1024 * 1024);
				indexing_time = timer.elapsed();
				std::cout << "SRP Building time:" << indexing_time << "  seconds.\n";
				FILE* fp = nullptr;
				fopen_s(&fp, "./indexes/maria_info.txt", "a");
				if (fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), memf - mem, indexing_time);
				saveIndex();
			}
			else {
				//in.close();
				std::cout << "Loading index from " << index_file << ":\n";
				float mem = (float)getCurrentRSS() / (1024 * 1024);
				loadIndex();
				float memf = (float)getCurrentRSS() / (1024 * 1024);
				std::cout << "Actual memory usage: " << memf - mem << " Mb \n";

			}
		}

		void buildIndex() {
			std::cout << std::endl
				<< "START HASHING..." << std::endl
				<< std::endl;
			lsh::timer timer;

			std::cout << "SETTING HASH PARAMETER..." << std::endl;
			timer.restart();
			SetHash();
			std::cout << "SETTING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;

			std::cout << "COMPUTING HASH..." << std::endl;
			timer.restart();
			GetHash(data);
			std::cout << "COMPUTING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;

			std::cout << "BUILDING INDEX..." << std::endl;
			std::cout << "THERE ARE " << L << " " << K << "-D HASH TABLES." << std::endl;
			timer.restart();

			if (part_map.empty())
				GetTables();
			else
				GetTables(part_map);

			std::cout << "BUILDING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;
		}

		void SetHash()
		{
			rndAs = new float[S * dim];
			// hashpar.rndAs2 = new float* [S];

			std::mt19937 rng(int(std::time(0)));
			// std::mt19937 rng(int(0));
			std::normal_distribution<float> nd;
			for (int i = 0; i < S * dim; ++i)
				rndAs[i] = (nd(rng));
		}

		void GetHash(Data& data)
		{
#if defined(__GNUC__) && defined(USE_BLAS)
			int m = hashvals.size();
			int k = dim;
			int n = S;

			float* A = data.base;
			float* B = rndAs;
			float* C = new float[m * n];

			memset(C, 0.0f, m * n * sizeof(float));
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				m, n, k, 1.0, A, k, B, k, 0.0, C, n);

			for (int i = 0; i < hashvals.size(); ++i)
			{
				hashvals[i].resize(L, 0);
				for (int j = 0; j < L; ++j)
				{
					for (int l = 0; l < K; ++l)
					{
						float val = C[i * S + j * K + l];
						// cal_inner_product(data[i],rndAs+(j*K+l)*dim,dim);
						if (val > 0)
							hashvals[i][j] |= (1 << l);
					}
				}
			}
#else

#pragma omp parallel for schedule(dynamic, 256)
			for (int i = 0; i < hashvals.size(); ++i)
			{
				hashvals[i].resize(L, 0);
				for (int j = 0; j < L; ++j)
				{
					for (int l = 0; l < K; ++l)
					{
						float val = cal_inner_product(data[i], rndAs + (j * K + l) * dim, dim);
						if (val > 0)
							hashvals[i][j] |= (1 << l);
					}
				}
			}
#endif

			// for(int i=0;i<10;++i){
			// 	for(int j=0;j<L;++j){
			// 		std::cout<<hashvals[i][j]<<" ";

			// 	}
			// 	std::cout<<std::endl;
			// }
		}

		void GetTables(std::vector<std::vector<int>>& part_map)
		{
			int num_parti = part_map.size();
			hash_tables.resize(num_parti * L);
			for (int i = 0; i < num_parti; ++i)
			{
				auto& part = part_map[i];
				// for (auto& id : part) {
				for (int l = 0; l < part.size(); ++l)
				{
					int id = part[l];
					for (int j = 0; j < L; ++j) {
						hash_tables[i * L + j].emplace_back(l, hashvals[id][j]);
					}
				}
			}

			for (auto& table : hash_tables)
			{
				std::sort(table.begin(), table.end());
			}
		}

		void GetTables()
		{
			hash_tables.resize(L);
			for (int i = 0; i < hashvals.size(); ++i)
			{
				int id = i;
				for (int j = 0; j < L; ++j)
				{
					hash_tables[j].emplace_back(id, hashvals[id][j]);
				}
			}

			for (auto& table : hash_tables)
			{
				std::sort(table.begin(), table.end());
			}
		}

		void saveIndex() {

			std::string file = index_file;
			std::ofstream out(file, std::ios::binary);

			out.write((char*)(&L), sizeof(int));
			out.write((char*)(&K), sizeof(int));
			out.write((char*)(&dim), sizeof(int));
			S = L * K;
			//save hashpar
			out.write((char*)(rndAs), sizeof(float) * S * dim);

			//save hashvals
			int N = hashvals.size();
			out.write((char*)(&N), sizeof(int));
			for (int i = 0;i < N;++i) {
				out.write((char*)(hashvals[i].data()), sizeof(uint16_t) * L);
			}

			//save hash tables
			int ntb = hash_tables.size();
			out.write((char*)(&ntb), sizeof(int));
			for (int j = 0; j < ntb; ++j) {
				int np = hash_tables[j].size();
				out.write((char*)(&np), sizeof(int));
				out.write((char*)(hash_tables[j].data()), sizeof(srpPair) * np);
			}
		}

		void loadIndex() {

			std::string file = index_file;
			std::ifstream in(file, std::ios::binary);

			in.read((char*)(&L), sizeof(int));
			in.read((char*)(&K), sizeof(int));
			in.read((char*)(&dim), sizeof(int));
			S = L * K;

			//load hashpar
			rndAs = new float[S * dim];
			in.read((char*)(rndAs), sizeof(float) * S * dim);

			//load hashvals
			int N = 0;
			in.read((char*)(&N), sizeof(int));
			hashvals.resize(N);
			for (int i = 0;i < N;++i) {
				hashvals[i].resize(L);
				in.read((char*)(hashvals[i].data()), sizeof(uint16_t) * L);
			}

			//load hash tables
			int ntb = 0;
			in.read((char*)(&ntb), sizeof(int));
			hash_tables.resize(ntb);
			for (int j = 0; j < ntb; ++j) {
				int np = 0;
				in.read((char*)(&np), sizeof(int));
				hash_tables[j].resize(np);
				in.read((char*)(hash_tables[j].data()), sizeof(srpPair) * np);
			}
		}

		void kjoin(std::vector<std::vector<Res>>& knns, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width)
			{
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			knns.resize(n);
			for (auto& nnset : knns) nnset.reserve(2 * width * L);

			for (int i = np * L; i < np * L + L; ++i)
			{
				auto& table = hash_tables[i];
				for (int j = 0; j < width; ++j)
				{
					for (int l = 0; l < j + width; ++l)
					{
						if (j != l)
							knns[table[j].id].emplace_back(table[l].id, 1.0f);
					}
				}
				for (int j = width; j < n - width; ++j)
				{
					for (int l = j - width; l < j + width; ++l)
					{
						if (j != l)
							knns[table[j].id].emplace_back(table[l].id, 1.0f);
					}
				}
				for (int j = n - width; j < n; ++j)
				{
					for (int l = j - width; l < n; ++l)
					{
						if (j != l)
							knns[table[j].id].emplace_back(table[l].id, 1.0f);
					}
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
				pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < knns.size(); ++i)
			{
				auto& pool = knns[i];
				for (auto& x : pool)
				{
					x.dist = calInnerProductReverse(data[ids[x.id]], data[ids[i]], dim);
#if defined(COUNT_CC)
					cost++;
#endif
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				std::sort(pool.begin(), pool.end());
				if (pool.size() > K)
					pool.resize(K);
			}
		}

		void kjoin1(std::vector<std::vector<Res>>& knns, std::vector<int>& ids, int np, int K, int width)
		{
			int n = hash_tables[np * L].size();
			if (n < 2 * width) {
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}

			int lc = width * 2 + 1;
			knns.resize(n, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));

#pragma omp parallel for
			for (int i = np * L; i < np * L + L; ++i) {
				auto& table = hash_tables[i];
				int bias = (i - np * L) * lc + width;
				for (int j = 0; j < width; ++j)
				{
					for (int l = 0; l < j; ++l)
					{
						{
							float inp = calInnerProductReverse(data[ids[table[j].id]],
								data[table[l].id], dim);
#if defined(COUNT_CC)
							cost++;
#endif
							knns[table[j].id][j - l + bias] = Res(table[l].id, inp);
							knns[table[l].id][l - j + bias] = Res(table[j].id, inp);
							// knns[table[j].id].emplace_back(table[l].id, inp);
							// knns[table[l].id].emplace_back(table[j].id, inp);
						}
					}
				}

#pragma omp parallel for schedule(dynamic, 256)
				for (int j = width; j < n; ++j)
				{
					for (int l = j - width; l < j; ++l)
					{
						{
							float inp = calInnerProductReverse(data[ids[table[j].id]],
								data[table[l].id], dim);
#if defined(COUNT_CC)
							cost++;
#endif
							knns[table[j].id][j - l + bias] = Res(table[l].id, inp);
							knns[table[l].id][l - j + bias] = Res(table[j].id, inp);
							// knns[table[j].id].emplace_back(table[l].id, inp);
							// knns[table[l].id].emplace_back(table[j].id, inp);
						}
					}
				}
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				std::sort(pool.begin(), pool.end());
				// if (pool.size() > K) pool.resize(K);
			}

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
				pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns)
			{
				// std::sort(pool.begin(), pool.end());
				if (pool.back().id == -1)
					pool.pop_back();
				if (pool.size() > K)
					pool.resize(K);
			}
		}

		//find NNS in np1 for the points in np2
		void kjoin(std::vector<std::vector<Res>>& knns, std::vector<int>& ids1, int np1,
			std::vector<int>& ids2, int np2, int K, int width)
		{
			int n1 = hash_tables[np1 * L].size();
			if (n1 < 2 * width)
			{
				std::cerr << "The hash table has not enough points!" << std::endl;
				return;
			}
			int n2 = hash_tables[np2 * L].size();
			int lc = width * 2 + 1;
			knns.resize(n2, std::vector<Res>(L * lc, Res(-1, FLT_MAX)));
			// knns.resize(n2);
			// for (auto& nnset : knns) nnset.reserve(2 * lc);

#pragma omp parallel for
			for (int i = 0; i < L; ++i) {
				auto& table1 = hash_tables[i + np1 * L];
				auto& table2 = hash_tables[i + np2 * L];
				int bias = i * lc;

				int pos1 = 0, pos2 = 0;

				while (pos2 < n2) {
					while (pos1 < n1 && table1[pos1].val < table2[pos2].val) pos1++;

					int start = std::max(pos1 - width, 0);
					int end = std::min(pos1 + width, n1 - 1);

					auto& vec2 = data[ids2[table2[pos2].id]];
					for (int j = start;j <= end;++j) {
						auto& vec1 = data[ids1[table1[j].id]];
						float dist = calInnerProductReverse(vec1, vec2, data.dim);
						knns[table2[pos2].id][bias + j - start] = Res(table1[j].id, dist);
					}
					pos2++;
				}
			}

			//#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) {
				std::sort(pool.begin(), pool.end());
				// if (pool.size() > K) pool.resize(K);
			}

			//#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) pool.erase(std::unique(pool.begin(), pool.end(), compareId), pool.end());

			//#pragma omp parallel for schedule(dynamic)
			for (auto& pool : knns) {
				if (pool.back().id == -1) pool.pop_back();
				if (pool.size() > K) pool.resize(K);
			}
		}

		void calQHash(queryN* q) {
			//hashvals[i].resize(L, 0);
			auto& vals = q->srpval;
			for (int j = 0; j < L; ++j)
			{
				for (int l = 0; l < K; ++l)
				{
					float val = cal_inner_product(q->queryPoint, rndAs + (j * K + l) * dim, dim);
					if (val > 0)
						vals[j] |= (1 << l);
				}
			}
		}

		void knn(queryN*& q) {
			int np = part_map.size() - 1;
			int cnt = 0;
			int ub = 200;
			std::vector<bool>& visited = q->visited;
			visited.resize(data.N, false);
			int size = part_map[np].size();
			if (part_map[np].size() < ub) {
				for (auto& u : part_map[np]) {
					visited[u] = true;
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
				}

				return;
			}

			int num_candidates = 0;
			uint32_t diff = 1;
			int lpos[4];
			int rpos[4];
			uint16_t lval[4], rval[4];
			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				rpos[i] = std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])) - table.begin();
				lpos[i] = rpos[i] - 1;
				while (lpos[i] >= 0 && table[lpos[i]].val >= q->srpval[i]) {
					lpos[i]--;
				}
				num_candidates += rpos[i] - lpos[i] - 1;
			}

			while (num_candidates < ub) {
				num_candidates = 0;
				for (int i = 0;i < L;++i) {
					auto& table = hash_tables[i + np * L];
					lval[i] = q->srpval[i] / diff * diff;
					rval[i] = lval[i] + diff;
					while (lpos[i] >= 0 && table[lpos[i]].val >= lval[i]) {
						lpos[i]--;
					}

					while (rpos[i] < size && table[rpos[i]].val <= rval[i]) {
						rpos[i]++;
					}
					num_candidates += rpos[i] - lpos[i] - 1;
				}
				diff *= 2;
			}

			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				for (int j = lpos[i] + 1;j < rpos[i];++j) {
					int u = part_map[np][table[j].id];
					if (visited[u]) continue;
					visited[u] = true;
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
					cnt++;
					if (cnt > ub) break;
				}
			}

			q->cost += ub;
		}

		int getEntryPoint(queryN*& q) {
			int np = part_map.size() - 1;
			int cnt = 0;
			int ub = 200;
			std::vector<bool>& visited = q->visited;
			visited.resize(data.N, false);
			int size = part_map[np].size();
			if (part_map[np].size() < ub) {
				for (auto& u : part_map[np]) {
					visited[u] = true;
					q->top_candidates.emplace(u, calInnerProductReverse(q->queryPoint, data[u], data.dim));
				}

				return;
			}

			int num_candidates = 0;
			uint32_t diff = 1;
			int lpos[4];
			int rpos[4];
			uint16_t lval[4], rval[4];
			for (int i = 0;i < L;++i) {
				auto& table = hash_tables[i + np * L];
				rpos[i] = std::upper_bound(table.begin(), table.end(), srpPair(-1, q->srpval[i])) - table.begin();
				lpos[i] = rpos[i] - 1;
				while (lpos[i] >= 0 && table[lpos[i]].val >= q->srpval[i]) {
					lpos[i]--;
				}
				num_candidates += rpos[i] - lpos[i] - 1;
			}

		}
	};
}
