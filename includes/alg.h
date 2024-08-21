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

extern std::atomic<size_t> _G_COST;

struct resOutput
{
	std::string algName;
	int L;
	int K;
	float c;
	float time;
	float recall;
	float ratio;
	float cost;
	float kRatio;
};

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

#if defined(unix) || defined(__unix__)
inline void localtime_s(tm* ltm, time_t* now) {}
#endif

template <typename mariaVx,typename Preprocess>
inline resOutput Alg0_maria(mariaVx& maria, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	
	Performance<queryN,Preprocess> perform;
	lsh::timer timer1;
	int t = 1;

	size_t cost1 = _G_COST;

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


	cost1 = _G_COST - cost1;

	resOutput res;
	res.algName = maria.alg_name;
	res.L = -1;
	res.K = m_;
	res.c = c_;
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)cost1) / ((long long)perform.num);
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

void saveAndShow(float c, int k, std::string& dataset, std::vector<resOutput>& res)
{
	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);
	std::string query_result = ("results/Running_result.txt");
	std::ofstream os(query_result, std::ios_base::app);
	os.seekp(0, std::ios_base::end); // 

	time_t zero_point = 1635153971 - 17 * 3600 - 27 * 60;//Let me set the time at 2021.10.25. 17:27 as the zero point
	size_t diff = (size_t)(now - zero_point);
#if defined(unix) || defined(__unix__)
	llt lt(diff);
#endif

	double date = ((float)(now - zero_point)) / 86400;
	float hour = date - floor(date);
	hour *= 24;
	float minute= hour = date - floor(date);


	std::stringstream ss;

	ss << "*******************************************************************************************************\n"
		<< "The result of FARGO for " << dataset << " is as follow: c="<<c<<", k="<<k
		<<"\n"
		<< "*******************************************************************************************************\n";

	ss << std::setw(12) << "algName"
		<< std::setw(12) << "c"
		<< std::setw(12) << "L"
		<< std::setw(12) << "K"
		<< std::setw(12) << "Time"
		<< std::setw(12) << "Recall"
		<< std::setw(12) << "Ratio"
		<< std::setw(12) << "Cost"
		<< std::endl
		<< std::endl;
	for (int i = 0; i < res.size(); ++i) {
		ss << std::setw(12) << res[i].algName
			<< std::setw(12) << res[i].c
			<< std::setw(12) << res[i].L
			<< std::setw(12) << res[i].K
			<< std::setw(12) << res[i].time
			<< std::setw(12) << res[i].recall
			<< std::setw(12) << res[i].ratio
			<< std::setw(12) << res[i].cost
			<< std::endl;
	}
#if defined(unix) || defined(__unix__)
	ss << "\n******************************************************************************************************\n"
		<< "                                                                                    "
		<< lt.date << '-' << lt.h << ':' << lt.m << ':' << lt.s
		<< "\n******************************************************************************************************\n\n\n";
#else
	ss << "\n******************************************************************************************************\n"
		<< "                                                                                    "
		<< ltm->tm_mon + 1 << '-' << ltm->tm_mday << ' ' << ltm->tm_hour << ':' << ltm->tm_min
		<< "\n*****************************************************************************************************\n\n\n";
#endif
	std::cout << ss.str();
	os << ss.str();
	os.close();  delete []ltm;
	//delete[] ltm;
}