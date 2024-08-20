#pragma once
#include "./utils/StructType.h"
#include "./utils/basis.h"
#include "./utils/Preprocess.h"
#include "./utils/patch_ubuntu.h"

#include <fstream>
#include <omp.h>
#include <random>
#include <time.h>
#include <thread>
#include <unordered_set>
#include <set>

extern std::atomic<size_t> _G_COST;

namespace hcnngLite {
	using pii = std::pair<int, int>;

	inline int rand_int(const int& min, const int& max) {
		static thread_local std::mt19937* generator = nullptr;
		if (!generator) generator = new std::mt19937(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
		//if (!generator) generator = new mt19937(0);
		std::uniform_int_distribution<int> distribution(min, max);
		return distribution(*generator);
	}

	struct Edge {
		int v1, v2;
		float weight;
		Edge() {
			v1 = -1;
			v2 = -1;
			weight = -1;
		}
		Edge(int _v1, int _v2, float _weight) {
			v1 = _v1;
			v2 = _v2;
			weight = _weight;
		}
		bool operator<(const Edge& e) const {
			return weight < e.weight;
		}
		~Edge() { }
	};

	using Graph = std::vector<std::vector<Edge>>;

	struct DisjointSet {
		int* parent;
		int* rank;
		DisjointSet(int N) {
			parent = new int[N];
			rank = new int[N];
			for (int i = 0; i < N; i++) {
				parent[i] = i;
				rank[i] = 0;
			}
		}

		void _union(int x, int y) {
			int xroot = parent[x];
			int yroot = parent[y];
			int xrank = rank[x];
			int yrank = rank[y];
			if (xroot == yroot)
				return;
			else if (xrank < yrank)
				parent[xroot] = yroot;
			else {
				parent[yroot] = xroot;
				if (xrank == yrank)
					rank[xroot] = rank[xroot] + 1;
			}
		}
		int find(int x) {
			if (parent[x] != x)
				parent[x] = find(parent[x]);
			return parent[x];
		}

		~DisjointSet() {
			delete[] parent;
			delete[] rank;
		}
	};

	//template<typename dist_t>

	using DISTFUNC = float(*)(float*, float*, int);

	//inline auto dist_t = calInnerProductReverse;

	template <DISTFUNC dist_t>
	class hcnng {
		int minsize_cl = 500;
		int num_cl = 10;
		int max_mst_degree = 3;
		int mlc = 0;
		int num_isolated = 0;
		int nq=-1;
		lsh::progress_display* pd = nullptr;

		float indexing_time = 0.0f;
		float indexing_cost = 0.0f;
		

		std::vector<std::vector<Res>> nngraph;
		std::vector<omp_lock_t> locks;

		std::vector<int> visited;
	public:
		Data data;
		std::string alg_name = "hcnng";

		hcnng(std::string datasetName, Data& data_, std::string file_graph, std::string index_result,
			int minsize_cl_, int num_cl_, int max_mst_degree_, bool rebuilt = false) {
			minsize_cl = minsize_cl_;
			num_cl = num_cl_;
			max_mst_degree = max_mst_degree_;
			data = data_;
			visited.resize(data.N, -1);

			lsh::timer timer;

			if (rebuilt || !findIndex(file_graph)) {
				_G_COST = 0;
				buildGraph();
				indexing_time = timer.elapsed();
				indexing_cost = (float)_G_COST / data.N;
				write_graph(file_graph);
				_G_COST = 0;

				printf("%s:\nIndexingTime=%f s, cc=%f.\n\n", datasetName.c_str(), indexing_time, indexing_cost);
				FILE* fp = nullptr;
				fopen_s(&fp, index_result.c_str(), "a");
				if (fp) fprintf(fp, "%s:\nIndexingTime=%f s, cc=%f.\n\n", datasetName.c_str(), indexing_time, indexing_cost);
				fclose(fp);
			}
		}

		bool findIndex(std::string& file){
			std::ifstream in(file.c_str(), std::ios::binary);
			if (!in.is_open()) {
				return false;
			}
			in.close();
			return true;
		}

		void buildGraph() {
			size_t N = data.N;
			size_t estimatedCC = 2 * ceil(log(N / minsize_cl)) + N * (minsize_cl - 1);
			size_t report_every = estimatedCC / 50;
			size_t next_report = 0;
			next_report += report_every;
			//Graph G(N);

			nngraph.resize(N);
			auto& G = nngraph;

			//std::vector<omp_lock_t> locks(N);
			locks.resize(N);
			for (int i = 0; i < N; i++) {
				omp_init_lock(&locks[i]);
				G[i].reserve(max_mst_degree * num_cl);
			}
			
			printf("creating clusters...\n");

			std::vector<std::vector<pii>> tps(num_cl);
			using pr_lr = std::pair<int, pii>;
			std::vector<pr_lr> partis;
			std::vector<int*> ids(num_cl, nullptr);
			mlc = N / (256 / num_cl);
			if (mlc < minsize_cl) mlc = minsize_cl;

			std::vector<std::vector<int>> degs(num_cl, std::vector<int>(N, 0));

#pragma omp parallel for
			for (int i = 0; i < num_cl; i++) {
				int* idx_points = new int[N];
				for (int j = 0; j < N; j++)
					idx_points[j] = j;

				//create_LC(points, idx_points, 0, N - 1, G, mlc, locks, tps[i]);
				createLargeCluster(idx_points, 0, N - 1, tps[i]);
				printf("end BIG cluster %d\n", i);
				//delete[] idx_points;
				ids[i] = idx_points;

			}

			for (int i = 0; i < num_cl; i++) {
				int* idx_points = ids[i];
				//#pragma omp parallel for
				for (int j = 0; j < tps[i].size(); ++j) {
					auto& x = tps[i][j];
					partis.emplace_back(i, x);
				}
			}
			std::cout << "LC cost: " << _G_COST << std::endl;
			printf("\n\nBuilding...\n");
			pd = new lsh::progress_display((size_t)N * (size_t)num_cl);
#pragma omp parallel for
			for (int i = 0; i < partis.size(); ++i) {
				auto& prs = partis[i];
				auto cl_id = prs.first;
				pii& x = prs.second;
				int left = x.first;
				int right = x.second;
				createClusters(ids[cl_id], left, right, degs[cl_id]);
				//create_clusters(points, idx_points, left, right, G, minsize_cl, locks, max_mst_degree);
				//pd += right - left + 1;
			}

			for (int i = 0; i < num_cl; i++) {
				int* idx_points = ids[i];
				delete[] idx_points;

			}
			printf("sorting...\n");
			sort_edges();
			//print_stats_graph(G);
		}

		inline void sort_edges() {
#pragma omp parallel for
			for (int i = 0; i < data.N; i++)
				sort(nngraph[i].begin(), nngraph[i].end());
		}

		int bipartition(int* idx_points, int left, int right) {
			int x = rand_int(left, right);
			int y = rand_int(left, right);
			while (y == x) y = rand_int(left, right);

			int num_points = right - left + 1;
			std::vector<std::pair<float, int>> dx(num_points);
			std::vector<std::pair<float, int>> dy(num_points);
			std::unordered_set<int> taken;
			for (int i = 0; i < num_points; i++) {
				dx[i] = std::make_pair(cal_L2sqr(data[idx_points[x]], data[idx_points[left + i]], data.dim), idx_points[left + i]);
				dy[i] = std::make_pair(cal_L2sqr(data[idx_points[y]], data[idx_points[left + i]], data.dim), idx_points[left + i]);

				// dx[i] = std::make_pair(dist_t(data[idx_points[x]], data[idx_points[left + i]], data.dim), idx_points[left + i]);
				// dy[i] = std::make_pair(dist_t(data[idx_points[y]], data[idx_points[left + i]], data.dim), idx_points[left + i]);
			}
			std::sort(dx.begin(), dx.end());
			std::sort(dy.begin(), dy.end());
			int i = 0, j = 0, turn = rand_int(0, 1), p = left, q = right;

			//turn = 0;

			while (i < num_points || j < num_points) {
				if (turn == 0) {
					if (i < num_points) {
						//if (not_in_set(dx[i].second, taken)) 
						if (taken.find(dx[i].second) == taken.end()) {
							idx_points[p] = dx[i].second;
							taken.insert(dx[i].second);
							p++;
							turn = (turn + 1) % 2;
						}
						i++;
					}
					else {
						turn = (turn + 1) % 2;
					}
				}
				else {
					if (j < num_points) {
						if (taken.find(dy[j].second) == taken.end()) {//BUG24.08.15: Write dy[j] as dy[i] !!!
							idx_points[q] = dy[j].second;
							taken.insert(dy[j].second);
							q--;
							turn = (turn + 1) % 2;
						}
						j++;
					}
					else {
						turn = (turn + 1) % 2;
					}
				}
			}

			//dx.clear();
			//dy.clear();
			//taken.clear();
			//std::vector<std::pair<float, int> >().swap(dx);
			//std::vector<std::pair<float, int> >().swap(dy);

			return p;
		}

		void createLargeCluster(int* idx_points, int left, int right, std::vector<pii>& pairs) {
			int num_points = right - left + 1;

			if (num_points < mlc) {
				pairs.push_back(std::make_pair(left, right));
			}
			else {
				
				int p = bipartition(idx_points, left, right);
				createLargeCluster(idx_points, left, p - 1, pairs);
				createLargeCluster(idx_points, p, right, pairs);
			}
		}

		void createClusters(int* idx_points, int left, int right, std::vector<int>& mst_degrees) {
			int num_points = right - left + 1;

			if (num_points < minsize_cl) {
				createExactMST(idx_points, left, right, mst_degrees);
				(*pd) += num_points;
			}
			else {
				int p = bipartition(idx_points, left, right);
				createClusters(idx_points, left, p - 1, mst_degrees);
				createClusters(idx_points, p, right, mst_degrees);
			}
		}

		void createExactMST(int* idx_points, int left, int right, std::vector<int>& mst_degrees) {
			int N = right - left + 1;
			if (N == 1) {
				num_isolated++;
				printf("%d\n", num_isolated);
			}
			//float cost;
			std::vector<Edge> full;
			//Graph mst;
			full.reserve(N * (N - 1));
			for (int i = 0; i < N; i++) {
				for (int j = i + 1; j < N; j++) {
					//float dist = cal_L2sqr(data[idx_points[left + i]], data[idx_points[left + j]], data.dim);
					float dist = dist_t(data[idx_points[left + i]], data[idx_points[left + j]], data.dim);
					//full.push_back(Edge(idx_points[left + i], idx_points[left + j], dist));
					//full.push_back(Edge(idx_points[left + j], idx_points[left + i], dist));
					full.push_back(Edge(i, j, dist));
					full.push_back(Edge(j, i, dist));
				}
						
			}
			//std::tie(mst, cost) = 
			kruskal(idx_points + left, full, N, mst_degrees);
			//return mst;
		}

		bool check_in_neighbors(int u, std::vector<Res>& nns) {
			for (int i = 0; i < nns.size(); i++) {
				if (nns[i].id == u)
					return true;
			}	
			return false;
		}

		void kruskal(int* arr, std::vector<Edge>& edges, int N,std::vector<int>& mst_degrees) {
			sort(edges.begin(), edges.end());
			//Graph MST(N);
			auto& graph = nngraph;
			DisjointSet* disjset = new DisjointSet(N);
			float cost = 0;
			for (Edge& e : edges) {
				int p1 = arr[e.v1], p2 = arr[e.v2];
				if (disjset->find(e.v1) != disjset->find(e.v2) && mst_degrees[p1] < max_mst_degree && mst_degrees[p2] < max_mst_degree) {
					

					omp_set_lock(&locks[p1]);
					if (!check_in_neighbors(p2, graph[p1])) graph[p1].emplace_back(p2,e.weight);
					mst_degrees[p1]++;
					omp_unset_lock(&locks[p1]);

					omp_set_lock(&locks[p2]);
					if (!check_in_neighbors(p1, graph[p2])) graph[p2].emplace_back(p1, e.weight);
					mst_degrees[p2]++;
					omp_unset_lock(&locks[p2]);

					//MST[e.v1].push_back(e);
					//MST[e.v2].push_back(Edge(e.v2, e.v1, e.weight));
					disjset->_union(e.v1, e.v2);
					cost += e.weight;

				}
			}
			delete disjset;
		}

		inline void write_graph(std::string& path_file) {
			FILE* F;
			int N;
			//F = fopen(path_file.c_str(), "wb");
			fopen_s(&F, path_file.c_str(), "wb");
			N = data.N;
			for (int i = 0; i < N; i++) {
				int degree = nngraph[i].size();
				fwrite(&degree, sizeof(int), 1, F);
				fwrite(&(nngraph[i][0]), sizeof(Res), degree, F);
				int* aux = new int[degree];
				//for (int j = 0; j < degree; j++)
				//	aux[j] = nngraph[i][j].v2;
				fwrite(aux, sizeof(int), degree, F);
				delete[] aux;
			}
			fclose(F);
		}

		void knn(queryN* q) {
			nq++;
			knn(q, 0, q->k + 100);
		}

		void knn(queryN* q, int start, int ef) {
			int cost = 0;
			lsh::timer timer;
			std::priority_queue<Res> accessed_candidates, top_candidates;
			visited[start] = nq;
			float dist = dist_t(q->queryPoint, data[start], data.dim);
			cost++;
			accessed_candidates.emplace(start, -dist);
			top_candidates.emplace(start, dist);

			while (!accessed_candidates.empty()) {
				Res top = accessed_candidates.top();
				if (-top.dist > top_candidates.top().dist) break;
				accessed_candidates.pop();

				for (auto& u : nngraph[top.id]) {
					if (visited[u.id] == nq) continue;
					visited[u.id] = nq;
					dist = dist_t(q->queryPoint, data[u.id], data.dim);
					cost++;
					accessed_candidates.emplace(u.id, -dist);
					top_candidates.emplace(u.id, dist);
					if (top_candidates.size() > ef) top_candidates.pop();
				}
			}

			while (top_candidates.size() > q->k) top_candidates.pop();

			q->res.resize(q->k);
			int pos = q->k;
			while (!top_candidates.empty()) {
				q->res[--pos] = top_candidates.top();
				top_candidates.pop();
			}
			q->time_total = timer.elapsed();
		}
	};

}