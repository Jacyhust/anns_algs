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
std::string data_fold1 = data_fold, data_fold2 = data_fold + ("ANN/");
#else
std::string data_fold = "E:/Dataset_for_c/", index_fold = " ";
std::string data_fold1 = data_fold;
std::string data_fold2 = data_fold + ("MIPS/");
//std::string data_fold2 = data_fold + ("ANN/");
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

    float c = 0.9f;
    int L = 5;
    int K = 12;
    rnndescent::Matrix<float> base_data;

    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));
    Partition parti(c, prep);
    mariaV6 mariaV6(prep.data, parti, L, K);

    return 0;
}