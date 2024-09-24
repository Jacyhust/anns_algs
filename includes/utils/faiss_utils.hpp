#pragma once

#include <algorithm>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include <omp.h>
#include <iterator>
#include <mutex>

namespace rnndescent
{

    using LockGuard = std::lock_guard<std::mutex>;

    inline void gen_random(std::mt19937 &rng, int *addr, const int size, const int N)
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

    struct Neighbor
    {
        int id;
        float distance;
        bool flag;

        Neighbor() = default;
        Neighbor(int id, float distance, bool f)
            : id(id), distance(distance), flag(f) {}

        inline bool operator<(const Neighbor &other) const
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

        Nhood(int l, int s, std::mt19937 &rng, int N)
        {
            M = s;
            nn_new.resize(s * 2);
            gen_random(rng, nn_new.data(), (int)nn_new.size(), N);
        }

        Nhood &operator=(const Nhood &other)
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

        Nhood(const Nhood &other)
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
                if (id == pool[i].id)
                    return;
            }
            if (pool.size() < pool.capacity())
            {
                pool.push_back(Neighbor(id, dist, true));
                std::push_heap(pool.begin(), pool.end());
            }
            else
            {
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(id, dist, true);
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
}