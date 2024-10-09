#include <iostream>
#include <memory>
#include <chrono>

//#include "../includes/indexDescdent.hpp"
//#include "../includes/RNNDescent.h"
//#include "../includes/utils/io.hpp"
#include "../includes/utils/StructType.h"
#include "../includes/mariaRNN_new.h"
#include "../includes/utils/performance.h"

#if defined(unix) || defined(__unix__)
//std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold = "/home/xizhao/dataset/", index_fold = "./indexes/";
std::string data_fold1 = data_fold, data_fold2 = data_fold + ("MIPS/");
#else
std::string data_fold = "E:/Dataset_for_c/", index_fold = " ";
std::string data_fold1 = data_fold;
std::string data_fold2 = data_fold + ("MIPS/");
//std::string data_fold2 = data_fold + ("ANN/");
#endif

int main(int argc, char* argv[])
{
    // uint64_t arr64[2] = { 63,64 };
    // std::cout << "1st ele is:" << arr64[0] << std::endl;
    // std::cout << "1st ele is:" << arr64[1] << std::endl;
    // std::cout << "diffcnt is:" << bitCounts(arr64, arr64 + 1) << std::endl;
    // return 0;

    // Res* a = new Res[5];
    // a[1] = Res(1, 1.0);
    // std::vector<Res> b(a, a + 5);
    // std::cout << b[1].id << std::endl;
    // a[1].id = 2;
    // std::cout << b[1].id << std::endl;
    // std::cout << a << std::endl;
    // std::cout << b.data() << std::endl;
    // return 1;

    std::string dataset = "audio";
    if (argc > 1) {
        dataset = argv[1];
    }
    std::string argvStr[4];
    argvStr[1] = (dataset);
    argvStr[2] = (dataset + ".index");
    argvStr[3] = (dataset + ".bench_graph");

    std::cout << "Using MARIA for " << argvStr[1] << std::endl;

    rnndescent::rnn_para para;
    para.S = 36;
    para.T1 = 3;
    para.T2 = 8;

    float c = 0.9f;
    int L = 4;
    int K = 16;
    rnndescent::Matrix<float> base_data;

    Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));
    Partition parti(c, prep);
    //mariaV6 mariaV6(prep.data, parti, L, K);
    //mariaV7 mariaV7(prep.data, parti, L, K);
    mariaV8 mariaV8(prep.data, prep.SquareLen, index_fold + argvStr[2] + "_maria", parti, L, K);
    LiteMARIA lm(prep.data, index_fold + argvStr[2] + "_maria", parti);
    //mariaV8.showInfo();
    //lm.showInfo();
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

    queries.N = 100;
    int repeat = 10;
#if defined(_DEBUG) || defined(_MSC_VER)
    repeat = 1;
#endif // _DEBUG
    int nq = queries.N * repeat;

    std::vector<queryN> qs;
    qs.reserve(nq);
    //queryN** qs = new queryN * [nq];
    std::cout << "nq= " << nq << std::endl;
    for (int i = 0; i < nq; ++i) {
        //qs[i] = new queryN(i % (queries.N), c_, k_, queries[i % (queries.N)], queries.dim, 1.0f);
        //qs[i] = new queryN(0, c_, k_, prep.data[i], queries.dim, 1.0f);
        qs.emplace_back(i % (queries.N), c_, k_, queries[i % (queries.N)], queries.dim, 1.0f);
    }

    // for(auto&x:nngraph){
    //   if(x.size()>M)x.resize(M);
    // }
    std::vector<resOutput> res;

    //res.push_back(searchFunction(mariaV6, qs, prep));
    //res.push_back(searchFunction(mariaV7, qs, prep));
    //res.push_back(searchFunction(mariaV8, qs, prep));
    //res.push_back(searchFunctionFn(mariaV8, qs, prep, 1));
    res.push_back(searchFunction(lm, qs, prep));
    res.push_back(searchFunctionFn(lm, qs, prep, 1));
    res.push_back(searchFunctionFn(lm, qs, prep, 2));
    res.push_back(searchFunctionFn(lm, qs, prep, 3));
    //res.push_back(searchFunctionFn(lm, qs, prep, 4));
    res.push_back(searchFunctionFn(lm, qs, prep, 6));
    //res.push_back(searchFunctionFn(lm, qs, prep, 2));
    saveAndShow(c, k_, dataset, res);

    return 0;
}