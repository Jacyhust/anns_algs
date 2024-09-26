#include <iostream>
#include <memory>
#include <chrono>

#include "../includes/RNNDescent.h"
#include "../includes/utils/io.hpp"

#include "sol.h"
#include "../includes/alg.h"
#include "../includes/indexDescdent.hpp"

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

    
    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));

    rnndescent::Matrix<float> base_data;
    base_data.load(prep.data.base, prep.data.N, prep.data.dim);
    rnndescent::MatrixOracle<float, rnndescent::metric::l2sqr> oracle(base_data);
    std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, para));

    bool rebuilt=0;
    if (argc > 2) rebuilt = std::stoi(argv[2]);
    std::string path="indexes/"+dataset+".rnnd";
    std::vector<std::vector<unsigned>> index_graph;

    if(rebuilt||!loadKNNG(index_graph,path)){
        auto start = chrono::high_resolution_clock::now();
        index->build(oracle.size(), true);
        auto end = chrono::high_resolution_clock::now();

        cout << "Elapsed time in milliseconds: "
            << 1.0 * std::chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000
            << " s" << endl;

        //std::cout << "sav_pth = " << sav_pth << "\n";
        
        index->extract_index_graph(index_graph);

        stat(index_graph);

        saveKNNG(index_graph,path);
    }

    

    //IO::saveBinVec(sav_pth, index_graph);
    indexFromKNNG indexRNN(index_graph, 0);
    test(indexRNN, prep.queries, prep);
    return 0;
}

