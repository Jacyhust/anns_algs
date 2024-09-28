#pragma once
#include <string>
// #include "Preprocess.h"
// //#include "mf_alsh.h"
// #include "performance.h"
// #include "basis.hpp"
// #include "hcnngLite.h"
// // #include "hnswlib.h"
// #include "maria.h"
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
#include <vector>
#include "./utils/patch_ubuntu.h"
#include "./utils/performance.h"
#include "./utils/basis.h"
//extern std::atomic<size_t> _G_COST;



extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;



template <typename algorithm,typename Preprocess>
inline resOutput Alg0_maria(algorithm& maria, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	
	Performance<queryN,Preprocess> perform;
	lsh::timer timer1;
	int t = 1;

	//size_t cost1 = _G_COST;

	lsh::progress_display pd(Qnum*t);
	for (int j = 0; j < Qnum*t; j++)
	{
		queryN query(j / t, c_, k_, prep, m_);
		maria.knn(&query);
		perform.update(query, prep);
		++pd;
	}

	float mean_time = (float)perform.time_total / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k_) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);


	//cost1 = _G_COST - cost1;

	resOutput res;
	res.algName = maria.alg_name;
	res.L = -1;
	res.K = m_;
	res.c = c_;
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)0) / ((long long)perform.num);
	res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}

#if defined(unix) || defined(__unix__)
//std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold1 = data_fold, data_fold2 = data_fold+("ANN/");
#else
std::string data_fold = "E:/Dataset_for_c/", index_fold = " ";
std::string data_fold1 = data_fold;
//std::string data_fold2 = data_fold + ("MIPS/");
std::string data_fold2 = data_fold + ("ANN/");
#endif

#if defined(unix) || defined(__unix__)
struct llt
{
	int date, h, m, s;
	llt(size_t diff) { set(diff); }
	void set(size_t diff)
	{
		date = diff / 86400;
		diff = diff % 86400;
		h = diff / 3600;
		diff = diff % 3600;
		m = diff / 60;
		s = diff % 60;
	}
};
#endif

