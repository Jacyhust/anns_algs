#pragma once

#include <vector>
#include <queue>

#include "utils/faiss_utils.hpp"
#include "utils/kgraph_utils.hpp"

namespace rnndescent
{

    struct rnn_para
    {
        unsigned T1{3};
        unsigned T2{20};
        unsigned S{16};
        unsigned R{96};
        unsigned K0{32};

        friend std::ostream &operator<<(std::ostream &os, const rnn_para &p)
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

    struct RNNDescent
    {
        IndexOracle const &matrixOracle;
        using storage_idx_t = int;

        using KNNGraph = std::vector<Nhood>;

        explicit RNNDescent(IndexOracle const &mo, rnn_para const &p);

        ~RNNDescent();

        void reset();

        /// Initialize the KNN graph randomly
        void init_graph();
        void update_neighbors();
        void build(const int n, bool verbose);

        void add_reverse_edges();

        void insert_nn(int id, int nn_id, float distance, bool flag);

        void extract_index_graph(std::vector<std::vector<unsigned>> &idx_graph);

        bool has_built = false;

        int T1 = 3;
        int T2 = 20;
        int S = 20;
        int R = 96;
        int K0 = 32; // maximum out-degree (mentioned as K in the original paper)

        int search_L = 0;       // size of candidate pool in searching
        int random_seed = 2021; // random seed for generators

        int d;     // dimensions
        int L = 8; // initial size of memory allocation

        int ntotal = 0;
        float alpha = 1.0;

        KNNGraph graph;
        std::vector<int> final_graph;
        std::vector<int> offsets;

        rnn_para para;

    };

} // namespace rnndescent