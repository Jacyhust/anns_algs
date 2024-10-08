#pragma once
#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <iterator>
#include <mutex>
#include "utils/basis.h"
#include "srp.h"
#include "utils/StructType.h"

#define MAXN (1<<30)

namespace rnnd
{

    struct rnn_para
    {
        unsigned T1{ 3 };
        unsigned T2{ 20 };
        unsigned S{ 16 };
        unsigned R{ 96 };
        unsigned K0{ 32 };

        friend std::ostream& operator<<(std::ostream& os, const rnn_para& p)
        {
            os << "rnn_para:\n"
                << "* T1       = " << p.T1 << "\n"
                << "* T2       = " << p.T2 << "\n"
                << "* S        = " << p.S << "\n"
                << "* R        = " << p.R << "\n"
                << "* K0       = " << p.K0 << "\n";
            return os;
        }
    };

    using LockGuard = std::lock_guard<std::mutex>;

    inline void gen_random(std::mt19937& rng, int* addr, const int size, const int N)
    {
        for (int i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (int i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        int off = rng() % N;
        for (int i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
    }

    inline void gen_random_rnn(std::mt19937& rng, int* addr, const int size, const int N)
    {
        for (int i = 0; i < size; ++i)
        {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (int i = 1; i < size; ++i)
        {
            if (addr[i] <= addr[i - 1])
            {
                addr[i] = addr[i - 1] + 1;
            }
        }
        int off = rng() % N;
        for (int i = 0; i < size; ++i)
        {
            addr[i] = (addr[i] + off) % N;
        }
    }

    //Revised by Xi: Oct 8, 2024
    //Use id to store id and flag
    //If id<MAXN: id=id,flag=true
    //Otherwise:id=id % MAXN,flag=false
    struct Neighbor
    {
        private:
        int id;

        //bool flag;

        public:
        float distance;
        Neighbor() = default;
        // Neighbor(int id, float distance, bool f)
        //     : id(id), distance(distance), flag(f) {}

        Neighbor(int id, float distance)
            : id(id), distance(distance) {}

        Neighbor(int id, float distance, int f)
            : id(id | ((1 - f) * MAXN)), distance(distance) {}

        int getId() {
            return id % MAXN;
        }

        bool getFlag() {
            return id < MAXN;
        }

        void setFalse() {
            //id = id + MAXN * (1 - f);
            id |= MAXN;
        }

        void setTrue() {
            //id = id + MAXN * (1 - f);
            id &= (MAXN - 1);
        }

        inline bool operator<(const Neighbor& other) const
        {
            return distance < other.distance;
        }
    };

    struct Nhood
    {
        std::mutex lock;
        std::vector<Neighbor> pool; // candidate pool (a max heap)
        int M;                      // number of new neighbors to be operated

        std::vector<int> nn_old;  // old neighbors
        std::vector<int> nn_new;  // new neighbors
        std::vector<int> rnn_old; // reverse old neighbors
        std::vector<int> rnn_new; // reverse new neighbors

        Nhood() = default;

        Nhood(int l, int s, std::mt19937& rng, int N)
        {
            M = s;
            nn_new.resize(s * 2);
            gen_random(rng, nn_new.data(), (int)nn_new.size(), N);
        }

        Nhood& operator=(const Nhood& other)
        {
            M = other.M;
            std::copy(
                other.nn_new.begin(),
                other.nn_new.end(),
                std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
            return *this;
        }

        Nhood(const Nhood& other)
        {
            M = other.M;
            std::copy(
                other.nn_new.begin(),
                other.nn_new.end(),
                std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
        }

        void insert(int id, float dist)
        {
            std::lock_guard<std::mutex> guard(lock);
            if (dist > pool.front().distance)
                return;
            for (int i = 0; i < pool.size(); i++)
            {
                if (id == pool[i].getId())
                    return;
            }
            if (pool.size() < pool.capacity())
            {
                pool.push_back(Neighbor(id, dist));
                std::push_heap(pool.begin(), pool.end());
            }
            else
            {
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(id, dist);
                std::push_heap(pool.begin(), pool.end());
            }
        }

        template <typename C>
        void join(C callback) const
        {
            for (int const i : nn_new)
            {
                for (int const j : nn_new)
                {
                    if (i < j)
                    {
                        callback(i, j);
                    }
                }
                for (int j : nn_old)
                {
                    callback(i, j);
                }
            }
        }
    };


    //using dist_t = cal_inner_product(const float*, const float*, int);
    using DISTFUNC = float(*)(const float*, const float*, int);

    //template <DISTFUNC dist_t>

//#define dist_t cal_inner_product
#define dist_t calInnerProductReverse

    struct RNNDescent
    {
        //IndexOracle const& matrixOracle;
        using storage_idx_t = int;

        using KNNGraph = std::vector<Nhood>;

        Data data;
        std::atomic<size_t> cost{ 0 };

        explicit RNNDescent(Data& data_, rnn_para const& para) {
            data = data_;
            ntotal = data.N;
            T1 = para.T1;
            T2 = para.T2;
            S = para.S;
            R = para.R;
            K0 = para.K0;
        }

        ~RNNDescent() {}

        void reset() {
            has_built = false;
            ntotal = 0;
            final_graph.resize(0);
            offsets.resize(0);
        }

        /// Initialize the KNN graph randomly
        void init_graph()
        {
            graph.reserve(ntotal);
            {
                std::mt19937 rng(random_seed * 6007);
                for (int i = 0; i < ntotal; i++)
                {
                    graph.emplace_back(L, S, rng, (int)ntotal);
                    // graph.push_back(Nhood(L, S, rng, (int)ntotal));
                }
            }

#pragma omp parallel
            {
                std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
#pragma omp for
                for (int i = 0; i < ntotal; i++)
                {
                    std::vector<int> tmp(S);

                    gen_random_rnn(rng, tmp.data(), S, ntotal);

                    for (int j = 0; j < S; j++)
                    {
                        int id = tmp[j];
                        if (id == i)
                            continue;


                        //float dist = matrixOracle(i, id);
                        float dist = dist_t(data[i], data[id], data.dim);
                        cost++;
                        graph[i].pool.emplace_back(id, dist, true);
                    }
                    std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
                    graph[i].pool.reserve(L);
                }
            }
        }

        /// Initialize the KNN graph randomly
        void init_graph(std::vector<std::vector<Res>>& init_nns)
        {
            graph.reserve(ntotal);
            {
                std::mt19937 rng(random_seed * 6007);
                for (int i = 0; i < ntotal; i++)
                {
                    graph.emplace_back(L, S, rng, (int)ntotal);
                    // graph.push_back(Nhood(L, S, rng, (int)ntotal));
                }
            }

#pragma omp parallel
            {
                std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
#pragma omp for
                for (int i = 0; i < ntotal; i++) {
                    for (auto& x : init_nns[i]) {
                        graph[i].pool.emplace_back(x.id, x.dist, true);
                        std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
                        graph[i].pool.reserve(L);
                    }
                    std::vector<Res>().swap(init_nns[i]);
                }
                //for (int i = 0; i < ntotal; i++)
                //{
                //    std::vector<int> tmp(S);

                //    gen_random_rnn(rng, tmp.data(), S, ntotal);

                //    for (int j = 0; j < S; j++)
                //    {
                //        int id = tmp[j];
                //        if (id == i)
                //            continue;


                //        //float dist = matrixOracle(i, id);
                //        float dist = dist_t(data[i], data[id], data.dim);

                //        graph[i].pool.emplace_back(id, dist, true);
                //    }
                //    std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
                //    graph[i].pool.reserve(L);
                //}
            }
        }

        void update_neighbors() {
#pragma omp parallel for schedule(dynamic, 256)
            for (int u = 0; u < ntotal; ++u)
            {
                auto& nhood = graph[u];
                auto& pool = nhood.pool;
                std::vector<Neighbor> new_pool;
                std::vector<Neighbor> old_pool;
                {
                    std::lock_guard<std::mutex> guard(nhood.lock);
                    // old_pool = pool;
                    // pool.clear();
                    old_pool.swap(pool);
                    //old_
                }
                std::sort(old_pool.begin(), old_pool.end());
                old_pool.erase(std::unique(old_pool.begin(), old_pool.end(),
                    [](Neighbor& a,
                        Neighbor& b)
                    {
                        return a.getId() == b.getId();
                    }),
                    old_pool.end());

                for (auto&& nn : old_pool)
                {
                    bool ok = true;
                    for (auto&& other_nn : new_pool)
                    {
                        // if (!nn.flag && !other_nn.flag)
                        // {
                        //     continue;
                        // }
                        if (!nn.getFlag() && !other_nn.getFlag()) {
                            continue;
                        }
                        if (nn.getId() == other_nn.getId())
                        {
                            ok = false;
                            break;
                        }
                        //float distance = matrixOracle(nn.id, other_nn.id);

                        float distance = dist_t(data[nn.getId()], data[other_nn.getId()], data.dim);
#if defined(COUNT_CC)
                        cost++;
#endif
                        if (distance < nn.distance)
                        {
                            ok = false;
                            insert_nn(other_nn.getId(), nn.getId(), distance, true);
                            break;
                        }
                    }
                    if (ok)
                    {
                        new_pool.emplace_back(nn);
                    }
                }

                for (auto&& nn : new_pool)
                {
                    //nn.getFlag() = false;
                    nn.setFalse();
                }
                {
                    std::lock_guard<std::mutex> guard(nhood.lock);
                    pool.insert(pool.end(), new_pool.begin(), new_pool.end());
                }
            }
        }


        void build(const int n, bool verbose) {
            if (verbose)
            {
                printf("Parameters: S=%d, R=%d, T1=%d, T2=%d\n", S, R, T1, T2);
            }

            ntotal = n;
            init_graph();

            for (int t1 = 0; t1 < T1; ++t1)
            {
                if (verbose)
                {
                    std::cout << "Iter " << t1 << " : " << std::flush;
                }
                for (int t2 = 0; t2 < T2; ++t2)
                {
                    update_neighbors();
                    if (verbose)
                    {
                        std::cout << "#" << std::flush;
                    }
                }

                if (t1 != T1 - 1)
                {
                    add_reverse_edges();
                }

                if (verbose)
                {
                    printf("\n");
                }
            }

#pragma omp parallel for
            for (int u = 0; u < n; ++u)
            {
                auto& pool = graph[u].pool;
                std::sort(pool.begin(), pool.end());
                pool.erase(std::unique(pool.begin(), pool.end(),
                    [](Neighbor& a,
                        Neighbor& b)
                    {
                        return a.getId() == b.getId();
                    }),
                    pool.end());
            }

            has_built = true;
        }

        void build(const int n, bool verbose, std::vector<std::vector<Res>>& init_nns) {
            if (verbose) {
                printf("Parameters: S=%d, R=%d, T1=%d, T2=%d\n", S, R, T1, T2);
            }

            ntotal = n;
            init_graph(init_nns);

            for (int t1 = 0; t1 < T1; ++t1)
            {
                if (verbose)
                {
                    std::cout << "Iter " << t1 << " : " << std::flush;
                }
                for (int t2 = 0; t2 < T2; ++t2)
                {
                    update_neighbors();
                    if (verbose)
                    {
                        std::cout << "#" << std::flush;
                    }
                }

                if (t1 != T1 - 1)
                {
                    add_reverse_edges();
                }

                if (verbose)
                {
                    printf("\n");
                }
            }

#pragma omp parallel for
            for (int u = 0; u < n; ++u)
            {
                auto& pool = graph[u].pool;
                std::sort(pool.begin(), pool.end());
                pool.erase(std::unique(pool.begin(), pool.end(),
                    [](Neighbor& a,
                        Neighbor& b)
                    {
                        return a.getId() == b.getId();
                    }),
                    pool.end());
            }

            has_built = true;
        }

        void add_reverse_edges() {
            std::vector<std::vector<Neighbor>> reverse_pools(ntotal);

#pragma omp parallel for
            for (int u = 0; u < ntotal; ++u)
            {
                for (auto&& nn : graph[u].pool)
                {
                    std::lock_guard<std::mutex> guard(graph[nn.getId()].lock);
                    reverse_pools[nn.getId()].emplace_back(u, nn.distance, nn.getFlag());
                }
            }

            //// new version
#pragma omp parallel for
            for (int u = 0; u < ntotal; ++u)
            {
                auto& pool = graph[u].pool;
                for (auto&& nn : pool)
                {
                    //nn.flag = true;
                    nn.setTrue();
                }
                auto& rpool = reverse_pools[u];
                rpool.insert(rpool.end(), pool.begin(), pool.end());
                pool.clear();
                std::sort(rpool.begin(), rpool.end());
                rpool.erase(std::unique(rpool.begin(), rpool.end(),
                    [](Neighbor& a,
                        Neighbor& b)
                    {
                        return a.getId() == b.getId();
                    }),
                    rpool.end());
                if (rpool.size() > R)
                {
                    rpool.resize(R);
                }
                pool.swap(rpool);
            }

#pragma omp parallel for
            for (int u = 0; u < ntotal; ++u)
            {
                auto& pool = graph[u].pool;
                std::sort(pool.begin(), pool.end());
                if (pool.size() > R)
                {
                    pool.resize(R);
                }
            }
        }

        void insert_nn(int id, int nn_id, float distance, bool flag) {
            auto& nhood = graph[id];
            {
                std::lock_guard<std::mutex> guard(nhood.lock);
                nhood.pool.emplace_back(nn_id, distance, flag);
            }
        }

        void extract_index_graph(std::vector<std::vector<unsigned>>& idx_graph) {
            auto n{ ntotal };
            //printf("n = %d\n", n);
            idx_graph.clear();
            idx_graph.resize(n);
#pragma omp parallel for
            for (int u = 0; u < n; ++u)
            {
                auto& pool = graph[u].pool;
                int K = std::min(K0, (int)pool.size());
                auto& nbhood = idx_graph[u];
                nbhood.reserve(K);
                for (int m = 0; m < K; ++m)
                {
                    int id = pool[m].getId();
                    nbhood.push_back(static_cast<unsigned>(id));
                }
            }
        }

        bool has_built = false;

        int T1 = 3;
        int T2 = 20;
        int S = 20;
        int R = 96;
        int K0 = 32; // maximum out-degree (mentioned as K in the original paper)

        int search_L = 0;       // size of candidate pool in searching
        int random_seed = 2021; // random seed for generators

        //int d;     // dimensions
        int L = 8; // initial size of memory allocation

        int ntotal = 0;
        float alpha = 1.0;

        KNNGraph graph;
        std::vector<int> final_graph;
        std::vector<int> offsets;

        //rnn_para para;

    };

} // namespace rnndescent