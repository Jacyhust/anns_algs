#include <iostream>
#include <memory>
#include <chrono>

//#include "../includes/indexDescdent.hpp"
//#include "../includes/RNNDescent.h"
//#include "../includes/utils/io.hpp"
#include "../includes/utils/StructType.h"

#include "../includes/mariaRNN.h"

#if defined(unix) || defined(__unix__)
//std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold1 = data_fold, data_fold2 = data_fold + ("MIPS/");
#else
std::string data_fold = "E:/Dataset_for_c/", index_fold = " ";
std::string data_fold1 = data_fold;
std::string data_fold2 = data_fold + ("MIPS/");
//std::string data_fold2 = data_fold + ("ANN/");
#endif

int main(int argc, char* argv[])
{
    std::string dataset = "audio2";
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

    float c = 0.9f;
    int L = 5;
    int K = 12;
    rnndescent::Matrix<float> base_data;

    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));
    Partition parti(c, prep);
    mariaV6 mariaV6(prep.data, parti, L, K);

    float c_ = 0.5;
    int k_ = 50;
    int M = 48;
    int recall = 0;
    float ratio = 0.0f;
    lsh::timer timer;
    auto times1 = timer.elapsed();
    lsh::timer timer11;
    int cost = 0;

    auto& queries = prep.queries;
    queryN** qs = new queryN * [queries.N];
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
        mariaV6.knn(qs[i]);
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

    return 0;
}