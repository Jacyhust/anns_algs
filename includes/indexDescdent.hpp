#include "alg.h"
#include <assert.h>

class indexFromKNNG
{
	std::vector<std::vector<uint32_t>> apg;
public:
	indexFromKNNG(std::vector<std::vector<uint32_t>>& knng, int M = 24) {
        if (M == 0) {
            apg.swap(knng);
            return;
        }

        std::cout << "Genarating APG form KNNG... " << std::endl;
        lsh::timer timer;
        int N = knng.size();
        apg.resize(N);
        //int M=24;
        std::vector<std::vector<uint32_t>> rnn(N);

        lsh::progress_display pd(N);
        for (int i = 0; i < N; ++i) {
            for (auto& u : knng[i]) {
                rnn[u].emplace_back(i);
            }
            ++pd;
        }

        std::cout << "Genarating rnn time: " << timer.elapsed() << " s." << std::endl;
        timer.restart();
        lsh::progress_display pd0(N);
        for (int i = 0; i < N; ++i) {
            apg[i].reserve(2 * M);
            for (int j = 0; j < M; ++j) {
                apg[i].emplace_back(knng[i][j]);
            }
            int M0 = M;
            if (M0 > rnn[i].size()) M0 = rnn[i].size();
            //M0=rnn[i].size();
            int l = rnn[i].size() - 1;
            for (int j = 0; j < M0; ++j) {
                apg[i].emplace_back(rnn[i][l - j]);
            }

            ++pd0;
        }

        std::cout << "Merging time: " << timer.elapsed() << " s." << std::endl;
        //knng.swap(apg);
	}

    void searchInKnng(Data& data, queryN* q, int start, int ef) {
        auto& nngraph = apg;
        int cost = 0;
        //std::cout<<"size of knng: "<<nngraph.size()<<std::endl;
        lsh::timer timer;
        std::priority_queue<Res> accessed_candidates, top_candidates;
        int n = nngraph.size();
        std::vector<bool> visited(n, false);
        visited[start] = true;
        float dist = cal_L2sqr(q->queryPoint, data[start], data.dim);
        cost++;
        accessed_candidates.emplace(start, -dist);
        top_candidates.emplace(start, dist);

        while (!accessed_candidates.empty()) {
            Res top = accessed_candidates.top();
            if (-top.dist > top_candidates.top().dist) break;
            accessed_candidates.pop();

            for (auto& u : nngraph[top.id]) {
                if (visited[u]) continue;
                visited[u] = true;
                dist = cal_L2sqr(q->queryPoint, data[u], data.dim);
                cost++;
                accessed_candidates.emplace(u, -dist);
                top_candidates.emplace(u, dist);
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
        q->cost = cost;
    }

    void searchInKnng(Data& data, queryN* q,
        std::vector<int>& ep_set, int ef) {
        auto& nngraph = apg;
        int cost = 0;
        //std::cout<<"size of knng: "<<nngraph.size()<<std::endl;
        lsh::timer timer;
        std::priority_queue<Res> accessed_candidates, top_candidates;
        int n = nngraph.size();
        std::vector<bool> visited(n, false);

        for (auto& start : ep_set) {
            visited[start] = true;
            float dist = cal_L2sqr(q->queryPoint, data[start], data.dim);
            cost++;
            accessed_candidates.emplace(start, -dist);
            top_candidates.emplace(start, dist);
        }

        while (!accessed_candidates.empty()) {
            Res top = accessed_candidates.top();
            if (-top.dist > top_candidates.top().dist) break;
            accessed_candidates.pop();

            for (auto& u : nngraph[top.id]) {
                if (visited[u]) continue;
                visited[u] = true;
                float dist = cal_L2sqr(q->queryPoint, data[u], data.dim);
                cost++;
                accessed_candidates.emplace(u, -dist);
                top_candidates.emplace(u, dist);
                if (top_candidates.size() > ef) top_candidates.pop();
            }

            // for (auto& u : nngraph[top.id]) {
            // 	if (visited[u]) continue;
            // 	visited[u] = true;
            // 	dist = cal_L2sqr(q->queryPoint, data[u], data.dim);
            // 	cost++;
            // 	accessed_candidates.emplace(u, -dist);
            // 	top_candidates.emplace(u, dist);
            // 	if (top_candidates.size() > ef) top_candidates.pop();
            // }

        }

        while (top_candidates.size() > q->k) top_candidates.pop();

        q->res.resize(q->k);
        int pos = q->k;
        while (!top_candidates.empty()) {
            q->res[--pos] = top_candidates.top();
            top_candidates.pop();
        }
        q->time_total = timer.elapsed();
        q->cost = cost;
    }
};

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knng a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void saveKNNG(const std::vector<std::vector<uint32_t>>& knng,
    const std::string& path = "output.bin") {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    //int K = 100;
    const uint32_t N = knng.size();
    std::cout << "Saving KNN Graph (" << knng.size() << " X 100) to " << path
        << std::endl;
    //  cout<<"knng.front().size()" << knng.front().size()<<"\n";
    //assert(knng.front().size() == K);

    ofs.write(reinterpret_cast<char const*>(&N), sizeof(uint32_t));
    //ofs.write(reinterpret_cast<char const *>(&K), sizeof(int));
    for (unsigned i = 0; i < knng.size(); ++i) {
        auto const& knn = knng[i];
        int K = knn.size();
        ofs.write(reinterpret_cast<char const*>(&K), sizeof(int));
        ofs.write(reinterpret_cast<char const*>(&knn[0]), K * sizeof(uint32_t));
    }
    ofs.close();
}

bool loadKNNG(std::vector<std::vector<uint32_t>>& knng, const std::string& path = "output.bin") {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    std::cout << "Loading kNNG from " << path << std::endl;

    uint32_t N;
    int K;

    // Read N and K
    ifs.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));


    // Resize the outer vector to hold N vectors
    knng.resize(N);

    // Read each vector
    for (uint32_t i = 0; i < N; ++i) {
        ifs.read(reinterpret_cast<char*>(&K), sizeof(int));
        knng[i].resize(K);
        ifs.read(reinterpret_cast<char*>(&knng[i][0]), K * sizeof(uint32_t));
    }

    ifs.close();
    return true;
}

void test(indexFromKNNG& index,Data& queries,Preprocess& prep) {
    float c_ = 0.5;
    int k_ = 50;
    int M = 48;
    int recall = 0;
    float ratio = 0.0f;
    lsh::timer timer;
    auto times1 = timer.elapsed();
    lsh::timer timer11;
    int cost = 0;

    queryN** qs = new queryN * [queries.N];
    queryN** qs0 = new queryN * [queries.N];
    queries.N = 100;
    std::cout << "nq= " << queries.N << std::endl;
    for (int i = 0; i < queries.N; ++i) {
        qs[i] = new queryN(0, c_, k_, queries[i], queries.dim, 1.0f);
    }

    // for(auto&x:nngraph){
    //   if(x.size()>M)x.resize(M);
    // }

    timer11.restart();
    for (int i = 0; i < queries.N; ++i) {
        //queryN q(0 , c_, k_, prep.queries[i],prep.queries.dim, 1.0f);
        std::vector<int> eps(prep.benchmark.indice[i] + 50, prep.benchmark.indice[i] + 80);
        //searchInKnng(nngraph, prep.data, qs[i], eps, k_ + 100);
        index.searchInKnng(prep.data, qs[i], eps, k_ + 100);
        //searchInKnng(nngraph, prep.data, qs[i], prep.benchmark.indice[i][0], k_+200);
        //searchInKnng(nngraph, prep.data, qs[i], 0, k_+100);
    }
    std::cout << "Query1 Time= " << (float)(timer11.elapsed() * 1000) / (prep.queries.N)
        << " ms." << std::endl;


    for (int i = 0; i < prep.queries.N; ++i) {
        cost += qs[i]->cost;
        for (int k = 0; k < k_; ++k) {
            ratio += sqrt(qs[i]->res[k].dist) / prep.benchmark.innerproduct[i][k];
            //ratio+=(q.res[k].dist)/prep.benchmark.indice[i][k];
            for (int l = 0; l < k_; ++l) {
                if (qs[i]->res[k].id == prep.benchmark.indice[i][l]) {
                    recall++;
                    break;
                }
            }
        }
    }



    auto times11 = timer.elapsed();
    std::cout << "Recall= " << (float)recall / (prep.queries.N * k_) << std::endl;
    std::cout << "Ratio = " << (float)ratio / (prep.queries.N * k_) << std::endl;
    std::cout << "Cost  = " << (float)cost / (prep.queries.N) << std::endl;
    std::cout << "Query1 Time= " << (float)(timer11.elapsed() * 1000) / (prep.queries.N) << " ms." << std::endl;

}