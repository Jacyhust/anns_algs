#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <cstring>
#include <chrono>

#include "../includes/hcnngLite.h"
#include "../includes/alg.h"
#include "sol.h"
//#include "mf_alsh.h"

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

//std::atomic<size_t> _G_COST=0;

int main(int argc, char const* argv[])
{
	// std::string dataset = "mnist";
	// if (argc > 1) {
	// 	dataset = argv[1];
	// }
	// std::string argvStr[4];
	// argvStr[1] = (dataset + ".data");
	// argvStr[2] = (dataset + ".index");
	// argvStr[3] = (dataset + ".bench_graph");
	
	// float c = 0.9f;
	// int k = 50;
	// int m, L, K;

	// std::cout << "Using FARGO for " << argvStr[1] << std::endl;
	// Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));
	// std::vector<resOutput> res;
	// m = 0;
	// L = 5;
	// K = 12;
	// c = 0.8;

	// //Parameter param(prep, L, K, 1);
	// lsh::timer timer;
	// Partition parti(c, prep);
	
	// int minsize_cl = 1500;
	// int num_cl = 10;
	// int max_mst_degree = 3;

	// hcnngLite::hcnng<calInnerProductReverse> hcnng(dataset, prep.data, data_fold2 + argvStr[2] + "_hcnng", "index_result.txt",
	// 	minsize_cl, num_cl, max_mst_degree, 1);

	// res.push_back(Alg0_maria(hcnng, c, 100, k, L, K, prep));
	// std::vector<int> ms = { 0,100,200,400,800,1200,1600,3200,6400};

	// saveAndShow(c, k, dataset, res);
	return 0;
}
