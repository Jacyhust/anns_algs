#include <iostream>
#include <memory>
#include <chrono>

#include "../includes/RNNDescent.h"
#include "../includes/utils/io.hpp"
//#include "sol.h"
#include "../includes/utils/StructType.h"
//#include "../includes/alg.h"
#include "../includes/indexDescdent.hpp"

#if defined(unix) || defined(__unix__)
//std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold1 = data_fold, data_fold2 = data_fold + ("ANN/");
#else
std::string data_fold = "E:/Dataset_for_c/", index_fold = " ";
std::string data_fold1 = data_fold;
//std::string data_fold2 = data_fold + ("MIPS/");
std::string data_fold2 = data_fold + ("ANN/");
#endif

int main(int argc, char* argv[])
{
    std::string dataset = "audio";
    if (argc > 1) {
        dataset = argv[1];
    }
    std::string argvStr[4];
    argvStr[1] = (dataset + ".data");
    argvStr[2] = (dataset + ".index");
    argvStr[3] = (dataset + ".bench_graph");

    std::cout << "Using MARIA for " << argvStr[1] << std::endl;

    rnndescent::rnn_para para;
    para.S = 36;
    para.T1 = 3;
    para.T2 = 8;

    rnndescent::Matrix<float> base_data;
    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));

    base_data.load(prep.data.base, prep.data.N, prep.data.dim);

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
    indexFromKNNG indexRNN(index_graph, 0);
    test(indexRNN, prep.queries, prep);
    return 0;
}