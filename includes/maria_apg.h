#pragma once
#include <mutex>
#include <algorithm>
#include "utils/Preprocess.h"
#include "srp.h"
#include "RNNDescent.h"
#include "rnnd.h"
#include <unordered_set>

class mariaV9
{
    private:
    std::string index_file;
    std::string index_srp;
    Partition& parti;
    //std::vector<std::vector<std::vector<uint32_t>>> knngs; // edges in each block
    //std::vector<std::vector<uint32_t>> tangential_lists;
    std::vector<std::vector<Res>> knng;
    std::vector<mp_mutex> locks;
    lsh::srp* srp = nullptr;
    rnnd::rnn_para para;
    int* link_lists = nullptr;
    Data data;
    std::atomic<size_t> cost{ 0 };
    int width = 20;
    int efC = 80;
    int M = 24;
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
    float* square_norms = nullptr;
    std::string alg_name = "mariaV8";
    // char* link_lists = nullptr;

    public:
    mariaV9(Data& data_, float* norms, const std::string& file, Partition& part_, int L_, int K_) : parti(part_)
    {
        data = data_;
        square_norms = norms;
        N = data.N;
        dim = data.dim;
        L = L_;
        K = K_;
        index_file = file + ".mariaV9";
        index_srp = file + ".srp";
        // const int min_size = 400;
        para.S = 36;
        para.T1 = 4;
        para.T2 = 4;

        lsh::timer timer;
        if (1 || !exists_test(index_file))
        {
            std::cout << "MARIAV9 Building ..." << "  seconds.\n";
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            buildIndex();
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            indexing_time = timer.elapsed();
            std::cout << "MARIAV9 Building time:" << indexing_time << "  seconds.\n";
            FILE* fp = nullptr;
            fopen_s(&fp, "./indexes/maria_info.txt", "a");
            if (fp)
                fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), memf - mem, indexing_time);
            saveIndex();
        }
        else
        {
            // in.close();
            srp = new lsh::srp(data, parti.EachParti, index_srp, data.N, data.dim);
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
        srp = new lsh::srp(data, parti.EachParti, index_srp, data.N, data.dim, L, K, 1);
        // return;
        //knngs.resize(parti.numChunks);
        lsh::timer timer;
        float time = 0.0f;

        //lsh::progress_display pd(N);
        knng.resize(N);
#ifdef COUNT_PD
        srp->resetPD();
#endif
        //#pragma omp parallel for schedule(dynamic)
        std::vector<int> bfC_block_id;
        for (int i = parti.numChunks - 1; i >= 0; --i) {
            // if (parti.EachParti[i].size() < 100) {
            // 	pd += parti.EachParti[i].size();
            // 	continue;
            // }
            if (parti.EachParti[i].size() < 256) {
                //bfConstruction(i, knng);
                bfC_block_id.emplace_back(i);
#ifdef COUNT_PD
                srp->updatePD(parti.EachParti[i].size());
#endif
                //pd += parti.EachParti[i].size();
                continue;
            }
            std::vector<std::vector<Res>> knns;
            srp->kjoin2_new(knns, parti.EachParti[i], i, para.S, width);
            for (int j = 0;j < knns.size();++j) {
                knns[j].resize(M);
                knng[parti.EachParti[i][j]].swap(knns[j]);
            }
            //pd += knns.size();
        }

        std::cout << "SRP Init TIME: " << timer.elapsed() << "s." << std::endl
            << std::endl;

        // rnnd::RNNDescent index(data, para);
        // index.build(data.N, 0, knng);
        // index.extract_index_graph(tangential_lists);

        // for (int i = parti.numChunks - 1; i >= 0; --i) {
        // 	if (parti.EachParti[i].size() < 256) {
        // 		bfConstruction(i, knng);
        // 		continue;
        // 	}
        // }

        update();

        for (auto& id : bfC_block_id) {
            bfConstruction(id, knng);

        }
        std::cout << "UPDATE TIME: " << timer.elapsed() << "s." << std::endl
            << std::endl;
        timer.restart();

        // std::vector<block_pairs> bps;
        auto& bps = conn_blocks;
        int SS = 32;
        // #pragma omp parallel for schedule(dynamic)
        size_t est_cost = 0;
        for (int i = parti.numChunks - 1; i >= 0; --i)
        {
            int init_S = SS;
            int j = 1;
            while (i - j >= 0)
            {
                // int init_K = (2 * init_S + 32) / L;
                int init_K = (2 * init_S + SS);
                bps.emplace_back(i - j, i, init_S, init_K);
                est_cost += parti.EachParti[i].size() * init_S * (2 * L + para.S);
                j *= 2;
                if (init_S > 1)
                    init_S /= 2;
            }
        }

        lsh::progress_display pd1(est_cost);
        //#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < bps.size(); ++i) {
            interConnection1(bps[i]);
            pd1 += parti.EachParti[bps[i].block2_id].size() * bps[i].S * (2 * L + para.S);
        }

        std::cout << "Inter-Connect TIME: " << timer.elapsed() << "s." << std::endl
            << std::endl;

        // std::cout << "CONSTRUCTING COST: " << (float)cost/N << std::endl;
        // cost+=srp.getCost();
        // std::cout << "SRP SEARCH COST (s): " << time << std::endl;
    }

    void update() {
        lsh::progress_display pd(knng.size());
#pragma omp parallel for schedule(dynamic)
        for (int i = 0;i < knng.size();++i) {
            searchInBuilding(i);
            ++pd;
        }
    }

    void searchInBuilding(int i) {
        if (knng[i].empty()) return;

        float* q = data[i];
        std::priority_queue<Res> top_candidates, candidate_set;
        //std::vector<bool> visited(knng1.size(), false);
        std::unordered_set<int> visited;
        for (auto& res : knng[i])
        {
            top_candidates.push(res);
            res.dist *= -1.0f;
            candidate_set.push(res);
            visited.emplace(res.id);
            //visited[res.id] = true;
        }

        while (top_candidates.size() > efC)
            top_candidates.pop();

        while (!candidate_set.empty()) {
            auto top = candidate_set.top();
            candidate_set.pop();
            if (-top.dist > top_candidates.top().dist)
                break;
            for (auto& us : knng[top.id]) {
                int u = us.id;
                if (visited.find(u) != visited.end()) continue;
                visited.emplace(u);

                float dist = cal_inner_product(q, data[u], dim);
                candidate_set.emplace(u, dist);
                top_candidates.emplace(u, -dist);
                if (top_candidates.size() > efC)
                    top_candidates.pop();
            }
        }

        while (top_candidates.size() > M) top_candidates.pop();

        {//For lock i
            write_lock(locks[i]);
            knng[i].clear();
            while (!top_candidates.empty()) {
                auto top = top_candidates.top();
                knng[i].emplace_back(top);
                top_candidates.pop();
            }
            std::make_heap(knng[i].begin(), knng[i].end());
        }

        {//For lock i's neighbor
            for (auto& us : knng[i]) {
                write_lock(locks[us.id]);
                knng[us.id].emplace_back(i, us.dist);
                std::push_heap(knng[us.id].begin(), knng[us.id].end());
            }
        }


    }

    void bfConstruction(int i, std::vector<std::vector<Res>>& knng)
    {
        int n = parti.EachParti[i].size();
        if (n <= 1) return;

        int len = sqrt(n);
        //std::vector<std::vector<Res>> nnset(n, std::vector<Res>(n, Res(-1, FLT_MAX)));
//#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < n; ++j) {
            for (int l = std::max(0, j - len); l < j; ++l) {
                float dist = calInnerProductReverse(data[parti.EachParti[i][j]], data[parti.EachParti[i][l]], data.dim);
                //nnset[j][l] = Res(l, dist);
                //nnset[l][j] = Res(j, dist);
                knng[parti.EachParti[i][j]].emplace_back(parti.EachParti[i][l], dist);
                knng[parti.EachParti[i][l]].emplace_back(parti.EachParti[i][j], dist);
            }
        }

        // #pragma omp parallel for schedule(dynamic)
        //for (int j = 0; j < n; ++j) {
        //	std::sort(knng[parti.EachParti[i][j]].begin(), knng[parti.EachParti[i][j]].end());
        //	if (knng[parti.EachParti[i][j]].size() > para.S)knng[parti.EachParti[i][j]].resize(para.S);
        //}
        // auto& apg = knngs[i];
        // int size = para.S;
        // if (size > n)
        // 	size = n - 1;
        // apg.resize(n, std::vector<uint32_t>(size));
        // for (int j = 0; j < n; ++j)
        // {
        // 	for (int l = 0; l < size; ++l)
        // 	{
        // 		apg[j][l] = nnset[j][l].id;
        // 	}
        // }
    }

    // np1<np2
    void interConnection1(block_pairs& bp)
    {
        std::vector<std::vector<Res>> knns;
        srp->kjoin(knns, parti.EachParti[bp.block1_id], bp.block1_id,
            parti.EachParti[bp.block2_id], bp.block2_id, bp.S, bp.S);

        //auto& knng1 = knngs[bp.block1_id];
        bp.normal_edges.resize(parti.EachParti[bp.block2_id].size());
        // std::vector<int> visited(knng2.size(), -1);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < knns.size(); ++i)
        {
            float* q = data[parti.EachParti[bp.block2_id][i]];
            std::priority_queue<Res> top_candidates, candidate_set;
            //std::vector<bool> visited(knng1.size(), false);
            std::unordered_set<int> visited_sets;
            //knns[i].resize()
            for (auto& res : knns[i]) {
                //if (visited[res.id]) continue;
                if (visited_sets.find(res.id) != visited_sets.end()) continue;
                visited_sets.emplace(res.id);
                //visited[res.id] = true;
                top_candidates.push(res);
                //res.dist *= -1.0f;
                //candidate_set.push(res);
                for (auto& us : knng[res.id]) {
                    int u = us.id;
                    if (visited_sets.find(res.id) != visited_sets.end()) continue;
                    visited_sets.emplace(res.id);
                    float dist = cal_inner_product(q, data[parti.EachParti[bp.block1_id][u]], dim);
                    //candidate_set.emplace(u, dist);
                    top_candidates.emplace(u, -dist);
                    if (top_candidates.size() > bp.S) top_candidates.pop();
                }
            }

            bp.normal_edges[i].reserve(top_candidates.size());
            for (int j = 0; j < top_candidates.size(); ++j)
            {
                auto& top = top_candidates.top();
                bp.normal_edges[i].push_back(top.id);
                top_candidates.pop();
            }
        }
    }

    void searchInKnng_new(std::vector<std::vector<Res>>& apg, queryN* q, int start, int ef)
    {
        auto& nngraph = apg;
        int cost = 0;
        std::priority_queue<Res> accessed_candidates;
        auto& top_candidates = q->top_candidates;
        int n = nngraph.size();
        //std::vector<bool> visited(n, false);
        auto& visited = q->visited;
        visited[start] = true;
        float dist = calInnerProductReverse(q->queryPoint, data[start], data.dim);
        cost++;
        accessed_candidates.emplace(start, -dist);
        top_candidates.emplace(start, dist);

        while (!accessed_candidates.empty())
        {
            Res top = accessed_candidates.top();
            if (-top.dist > top_candidates.top().dist)
                break;
            accessed_candidates.pop();

            for (auto& us : nngraph[top.id])
            {
                int u = us.id;
                if (visited[u])
                    continue;
                visited[u] = true;
                dist = calInnerProductReverse(q->queryPoint, data[u], data.dim);
                cost++;
                accessed_candidates.emplace(u, -dist);
                top_candidates.emplace(u, dist);
                if (top_candidates.size() > ef)
                    top_candidates.pop();
            }
        }

        while (top_candidates.size() > q->k)
            top_candidates.pop();
        q->cost += cost;
    }

    void knn(queryN* q)
    {
        lsh::timer timer;
        timer.restart();
        q->visited.resize(data.N, false);
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
                auto& top_candidates = q->top_candidates;
                for (auto& x : parti.EachParti[i])
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
            //auto& knng = tangential_lists;

            searchInKnng_new(knng, q, parti.EachParti[i][0], ef);

            // break;
        }

        auto& top_candidates = q->top_candidates;

        q->res.resize(q->k);
        int pos = q->k;
        while (!top_candidates.empty())
        {
            q->res[--pos] = top_candidates.top();
            top_candidates.pop();
        }

        q->time_total = timer.elapsed();
    }


    void knn1(queryN* q) {}
    void knn2(queryN* q) {}
    void knn3(queryN* q) {}
    void knn4(queryN* q) {}
    void knn5(queryN* q) {}
    void knn6(queryN* q) {}

    void compute_maxsize()
    {
        int i = parti.numChunks - 1;
        size_per_point += para.S;
        int init_S = 32;
        int j = 1;
        //size_per_point += para.S;
        while (i - j >= 0)
        {
            size_per_point += init_S;
            j *= 2;
            if (init_S > 1)
                init_S /= 2;
        }

        // To align with 64B
        size_per_point = ((size_per_point - 1) / 16 + 1) * 16;
        //size_per_point *= 10;
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
        for (int i = 0;i < N;++i) {
            int* v = link_lists + size_per_point * i;
            //memcpy((void*)(v), (void*)(srp->hashvals[i].data()), sizeof(uint64_t));//v->hashval has the same address with v
            memcpy(reinterpret_cast<void*>(v), reinterpret_cast<void*>(srp->hashvals[i]), sizeof(uint64_t));
            float* pvnorm = (float*)(v + 2);
            *pvnorm = sqrt(square_norms[i]);
            v[3] = 0;
            //int* pvsize = (int*)(v + 3);
            //*pvsize = 0;
        }

        std::cout << "INITIA  TIME: " << timer.elapsed() << "s." << std::endl;
        timer.restart();

        for (int i = 0;i < data.N;++i) {
            int* v = (link_lists + size_per_point * i);
            int* links = (v + 4);
            for (int j = 0; j < knng[i].size(); ++j) {
                links[(v[3])] = knng[i][j].id;
                v[3] = v[3] + 1;
            }
        }
        std::cout << "TANGEN  TIME: " << timer.elapsed() << "s." << std::endl;
        timer.restart();

        for (auto& bps : conn_blocks)
        {
            auto& ids1 = parti.EachParti[bps.block1_id];
            auto& ids2 = parti.EachParti[bps.block2_id];
            auto& knng = bps.normal_edges;
#pragma omp parallel for schedule(dynamic, 256)
            for (int j = 0; j < knng.size(); ++j)
            {
                int id1 = ids2[j];
                // vertex* v = (vertex*)(link_lists + size_per_point * id1);
                int* v = (link_lists + size_per_point * id1);
                int* links = (v + 4);
                int* size = ((int*)(v + 3));
                for (auto& u : knng[j])
                {
                    // v->links[v->size++] = ids2[u];

                    links[(*size)++] = ids1[u];
                    // auto& size = *((int*)(v + 12));
                    // links[size++] = ids2[u];
                }
            }
        }

        std::cout << "NORMAL TIME: " << timer.elapsed() << "s." << std::endl;
        timer.restart();

        std::string file = index_file;
        std::ofstream out(file, std::ios::binary);
        out.write((char*)(&N), sizeof(int));
        out.write((char*)(&size_per_point), sizeof(size_t));
        out.write((char*)(link_lists), sizeof(int) * size_per_point * N);

        float mem = size_per_point * N * 4;
        mem /= (1 << 30);
        std::cout << "size per p : " << size_per_point << std::endl;
        std::cout << "File size  : " << mem << "GB." << std::endl;
        std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl;
    }

    void loadIndex(const std::string& file)
    {
        std::ifstream in(file, std::ios::binary);
        if (!in.good())
        {
            std::cerr << "Cannot open file:" << file << std::endl;
        }
        in.read((char*)(&N), sizeof(int));
        in.read((char*)(&size_per_point), sizeof(size_t));
        link_lists = new int[size_per_point * N];
        in.read((char*)(link_lists), sizeof(int) * size_per_point * N);
    }

    void showInfo()
    {
        // std::cout << "This is the info of V8:" << std::endl;
        // auto knng = knngs[0];
        // auto ids = parti.EachParti[0];
        // for (int i = 0; i < 10; ++i)
        // {
        // 	int j = 0;
        // 	for (j = 0; i < N; ++j)
        // 	{
        // 		if (ids[j] == i)
        // 			break;
        // 	}

        // 	printf("point-%d has %d neighbors:\n", i, knng[j].size());

        // 	for (auto& x : knng[j]) {
        // 		printf("%d\t", ids[x]);
        // 	}
        // 	printf("\n");
        // }
    }

    ~mariaV9()
    {
        // for (int i = 0; i < parti.numChunks; ++i) {
        //	delete apgs[i];
        // }
        // delete[] apgs;
    }
};
