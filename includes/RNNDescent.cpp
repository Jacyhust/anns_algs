
#include "RNNDescent.h"
#include <iostream>

namespace rnndescent
{

    void gen_random_rnn(std::mt19937 &rng, int *addr, const int size, const int N)
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

    // Insert a new point into the candidate pool in ascending order
    int insert_into_pool(Neighbor *addr, int size,
                         Neighbor nn)
    {
        // find the location to insert
        int left = 0, right = size - 1;
        if (addr[left].distance > nn.distance)
        {
            memmove((char *)&addr[left + 1], &addr[left],
                    size * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance)
        {
            addr[size] = nn;
            return size;
        }
        while (left < right - 1)
        {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)
                right = mid;
            else
                left = mid;
        }
        // check equal ID

        while (left > 0)
        {
            if (addr[left].distance < nn.distance)
                break;
            if (addr[left].id == nn.id)
                return size + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)
            return size + 1;
        memmove((char *)&addr[right + 1], &addr[right],
                (size - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }

    RNNDescent::RNNDescent(IndexOracle const &mo, rnn_para const &p) : matrixOracle(mo), para(p)
    {
        ntotal = mo.size();
        T1 = para.T1;
        T2 = para.T2;
        S = para.S;
        R = para.R;
        K0 = para.K0;
    }

    RNNDescent::~RNNDescent()
    {
    }

    void RNNDescent::init_graph()
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
                    float dist = matrixOracle(i, id);
                    graph[i].pool.emplace_back(id, dist, true);
                }
                std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
                graph[i].pool.reserve(L);
            }
        }
    }

    void RNNDescent::insert_nn(int id, int nn_id, float distance, bool flag)
    {
        auto &nhood = graph[id];
        {
            std::lock_guard<std::mutex> guard(nhood.lock);
            nhood.pool.emplace_back(nn_id, distance, flag);
        }
    }

    void RNNDescent::update_neighbors()
    {
#pragma omp parallel for schedule(dynamic, 256)
        for (int u = 0; u < ntotal; ++u)
        {
            auto &nhood = graph[u];
            auto &pool = nhood.pool;
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
                                       [](Neighbor &a,
                                          Neighbor &b)
                                       {
                                           return a.id == b.id;
                                       }),
                           old_pool.end());

            for (auto &&nn : old_pool)
            {
                bool ok = true;
                for (auto &&other_nn : new_pool)
                {
                    if (!nn.flag && !other_nn.flag)
                    {
                        continue;
                    }
                    if (nn.id == other_nn.id)
                    {
                        ok = false;
                        break;
                    }
                    float distance = matrixOracle(nn.id, other_nn.id);
                    if (distance < nn.distance)
                    {
                        ok = false;
                        insert_nn(other_nn.id, nn.id, distance, true);
                        break;
                    }
                }
                if (ok)
                {
                    new_pool.emplace_back(nn);
                }
            }

            for (auto &&nn : new_pool)
            {
                nn.flag = false;
            }
            {
                std::lock_guard<std::mutex> guard(nhood.lock);
                pool.insert(pool.end(), new_pool.begin(), new_pool.end());
            }
        }
    }

    void RNNDescent::add_reverse_edges()
    {
        std::vector<std::vector<Neighbor>> reverse_pools(ntotal);

#pragma omp parallel for
        for (int u = 0; u < ntotal; ++u)
        {
            for (auto &&nn : graph[u].pool)
            {
                std::lock_guard<std::mutex> guard(graph[nn.id].lock);
                reverse_pools[nn.id].emplace_back(u, nn.distance, nn.flag);
            }
        }

//// new version
#pragma omp parallel for
        for (int u = 0; u < ntotal; ++u)
        {
            auto &pool = graph[u].pool;
            for (auto &&nn : pool)
            {
                nn.flag = true;
            }
            auto &rpool = reverse_pools[u];
            rpool.insert(rpool.end(), pool.begin(), pool.end());
            pool.clear();
            std::sort(rpool.begin(), rpool.end());
            rpool.erase(std::unique(rpool.begin(), rpool.end(),
                                    [](Neighbor &a,
                                       Neighbor &b)
                                    {
                                        return a.id == b.id;
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
            auto &pool = graph[u].pool;
            std::sort(pool.begin(), pool.end());
            if (pool.size() > R)
            {
                pool.resize(R);
            }
        }
    }

    void RNNDescent::build(const int n,
                           bool verbose)
    {
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
            auto &pool = graph[u].pool;
            std::sort(pool.begin(), pool.end());
            pool.erase(std::unique(pool.begin(), pool.end(),
                                   [](Neighbor &a,
                                      Neighbor &b)
                                   {
                                       return a.id == b.id;
                                   }),
                       pool.end());
        }

        has_built = true;
    }

    void RNNDescent::extract_index_graph(std::vector<std::vector<unsigned>> &idx_graph)
    {
        auto n{ntotal};
        printf("n = %d\n", n);
        idx_graph.clear();
        idx_graph.resize(n);
#pragma omp parallel for
        for (int u = 0; u < n; ++u)
        {
            auto &pool = graph[u].pool;
            int K = std::min(K0, (int)pool.size());
            auto &nbhood = idx_graph[u];
            for (int m = 0; m < K; ++m)
            {
                int id = pool[m].id;
                nbhood.push_back(static_cast<unsigned>(id));
            }
        }
    }

    void RNNDescent::reset()
    {
        has_built = false;
        ntotal = 0;
        final_graph.resize(0);
        offsets.resize(0);
    }

} // namespace rnndescent