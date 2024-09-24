#include <iostream>
#include <memory>
#include <chrono>

#include "RNNDescent.h"
#include "utils/io.hpp"

#include "sol.h"
#include "../includes/alg.h"


using namespace std;

void stat(const vector<vector<unsigned>> &graph)
{
    size_t max_edge = 0;
    size_t min_edge = graph.size();
    size_t avg_edge = 0;
    for (auto &nbhood : graph)
    {
        auto size = nbhood.size();
        max_edge = std::max(max_edge, size);
        min_edge = std::min(min_edge, size);
        avg_edge += size;
    }
    std::cout << "max_edge = " << max_edge << "\nmin_edge = " << min_edge << "\navg_edge = " << (1.0 * avg_edge / graph.size()) << "\n";
}

int main(int argc, char *argv[])
{

    std::string dataset = "audio";
	if (argc > 1) {
		dataset = argv[1];
	}
	std::string argvStr[4];
	argvStr[1] = (dataset + ".data");
	argvStr[2] = (dataset + ".index");
	argvStr[3] = (dataset + ".bench_graph");

    std::cout << "Using FARGO for " << argvStr[1] << std::endl;

    rnndescent::rnn_para para;
    para.S = 36;
    para.T1 = 3;
    para.T2 = 8;

    rnndescent::Matrix<float> base_data;
    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));

    base_data.load(prep.data.base,prep.data.N,prep.data.dim);
    //base_data.load(base_path, 128, 0, 4);



    // size_t data_size;
    // base_data.resize(data_size, 100);
    // for (unsigned id = 0; id < data_size; ++id) {
    //     base_data.add_test((float*)(data + (size_t)id * 416 + 8));
    // }

    rnndescent::MatrixOracle<float, rnndescent::metric::l2sqr> oracle(base_data);

    std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, para));

    auto start = chrono::high_resolution_clock::now();
    index->build(oracle.size(), true);
    auto end = chrono::high_resolution_clock::now();

    cout << "Elapsed time in milliseconds: "
         << 1.0 * std::chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000
         << " s" << endl;

    //std::cout << "sav_pth = " << sav_pth << "\n";
    std::vector<std::vector<unsigned>> index_graph;
    index->extract_index_graph(index_graph);

    stat(index_graph);

    //IO::saveBinVec(sav_pth, index_graph);

    return 0;
}