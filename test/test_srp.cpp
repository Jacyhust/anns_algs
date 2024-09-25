#include <iostream>
#include <memory>
#include <chrono>

#include "../includes/RNNDescent.h"
#include "../includes/utils/io.hpp"

#include "sol.h"
#include "../includes/alg.h"
#include "../includes/indexDescdent.hpp"
#include "../includes/srp.h"


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
    para.T1 = 2;
    para.T2 = 8;

    rnndescent::Matrix<float> base_data;
    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));

    int L=5;
    int K=12;
    std::vector<std::vector<int>> part_map;

    lsh::srp srp(prep.data,part_map,prep.data.N,
    prep.data.dim,L,K);


    
}

