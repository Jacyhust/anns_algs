#pragma once
#include <set>
#include <algorithm>
//#include "Preprocess.h"
#include <iterator>
#include <iostream>
#include "basis.h"
template <class Query, class Preprocess>
class Performance
{
	public:
	//cost
	int cost;
	//
	std::vector<int> costs;
	// times of query
	int num;
	//
	float time_total = 0.0f;
	//
	float time_hash = 0.0f;
	//
	float time_sift;
	//
	float time_verify;
	//number of exact NN
	int NN_num;
	//number of results
	int res_num;
	//
	float ratio;
	// k-th ratio
	float kRatio = 0.0f;
	public:
	Performance() {
		cost = 0;
		num = 0;
		time_total = 0;
		time_hash = 0;
		time_sift = 0;
		time_verify = 0;
		NN_num = 0;
		res_num = 0;
		ratio = 0;
	}
	//update the Performance for each query
	void update(Query& query, Preprocess& prep) {
		num++;
		cost += query.cost;
		time_sift += query.time_sift;
		time_verify += query.time_verify;
		time_total += query.time_total;

		int len = query.costs.size();
		if (costs.size() == 0)
			costs.resize(len);

		for (int i = 0; i < len; i++)
		{
			costs[i] += query.costs[i];
		}

		int num0 = query.res.size();
		if (num0 > query.k)
			num0 = query.k;
		res_num += num0;

		std::set<int> set1, set2;
		std::vector<int> set_intersection;
		set_intersection.clear();
		set1.clear();
		set2.clear();

		for (int j = 0; j < num0; j++)
		{
			float rate = fabs(query.res[j].dist / prep.benchmark.innerproduct[query.qid][j]);
			ratio += rate;

			if (j == num0 - 1) kRatio += rate;

			set1.insert(query.res[j].id);
			set2.insert((int)prep.benchmark.indice[query.qid][j]);
		}
		std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
			std::back_inserter(set_intersection));

		NN_num += set_intersection.size();
	}

	~Performance() {}
};


template <typename algorithm, typename queryN, typename Preprocess>
inline resOutput searchFunction(algorithm& alg, std::vector<queryN> qs, Preprocess& prep) {
	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	Performance<queryN, Preprocess> perform;
	int nq = qs.size();

	lsh::progress_display pd(nq);
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < nq; ++i) {
		alg.knn(&(qs[i]));
		++pd;
	}


	float tt = (float)(timer.elapsed() * 1000);
	std::cout << "Query Time= " << (float)(timer.elapsed() * 1000) << " ms." << std::endl;

	for (int i = 0; i < nq; ++i) {
		perform.update(qs[i], prep);
	}
	//for (int j = 0; j < Qnum * t; j++){
	//	queryN query(j / t, c_, k_, prep, m_);
	//	alg.knn(&query);
	//	perform.update(query, prep);
	//	++pd;
	//}

	//float mean_time = (float)perform.time_total / perform.num;
	resOutput res;
	res.algName = alg.alg_name;
	res.time = (float)perform.time_total / perform.num * 1000;
	res.qps = (float)nq / (tt / 1000);
	res.recall = ((float)perform.NN_num) / (perform.num * qs[0].k);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)perform.cost) / ((long long)perform.num);


	std::cout << "AVG QUERY TIME:    " << res.time << "ms." << std::endl << std::endl;
	std::cout << "AVG QPS:           " << res.qps << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << res.recall << std::endl;
	std::cout << "AVG RATIO:         " << res.ratio << std::endl;
	std::cout << "AVG COST:          " << res.cost << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);


	//cost1 = _G_COST - cost1;


	//res.L = -1;
	//res.K = m_;
	//res.c = c_;
	//res.time = mean_time * 1000;
	//res.recall = ((float)perform.NN_num) / (perform.num * k_);
	//res.ratio = ((float)perform.ratio) / (perform.res_num);
	//res.cost = ((float)0) / ((long long)perform.num);
	//res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}

template <typename algorithm, typename queryN, typename Preprocess>
inline resOutput searchFunctionFn(algorithm& alg, std::vector<queryN> qs, Preprocess& prep, int fn) {
	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	Performance<queryN, Preprocess> perform;
	int nq = qs.size();

	lsh::progress_display pd(nq);
	if (fn == 1) {
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < nq; ++i) {
			alg.knn1(&(qs[i]));
			++pd;
		}
	}
	else if (fn == 2) {
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < nq; ++i) {
			alg.knn2(&(qs[i]));
			++pd;
		}
	}
	else if (fn == 3) {
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < nq; ++i) {
			alg.knn3(&(qs[i]));
			++pd;
		}
	}
	else if (fn == 4) {
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < nq; ++i) {
			alg.knn1(&(qs[i]));
			++pd;
		}
	}



	float tt = (float)(timer.elapsed() * 1000);
	std::cout << "Query Time= " << (float)(timer.elapsed() * 1000) << " ms." << std::endl;

	for (int i = 0; i < nq; ++i) {
		perform.update(qs[i], prep);
	}
	//for (int j = 0; j < Qnum * t; j++){
	//	queryN query(j / t, c_, k_, prep, m_);
	//	alg.knn(&query);
	//	perform.update(query, prep);
	//	++pd;
	//}

	//float mean_time = (float)perform.time_total / perform.num;
	resOutput res;
	res.algName = alg.alg_name + std::to_string(fn);
	res.time = (float)perform.time_total / perform.num * 1000;
	res.qps = (float)nq / (tt / 1000);
	res.recall = ((float)perform.NN_num) / (perform.num * qs[0].k);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)perform.cost) / ((long long)perform.num);


	std::cout << "AVG QUERY TIME:    " << res.time << "ms." << std::endl << std::endl;
	std::cout << "AVG QPS:           " << res.qps << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << res.recall << std::endl;
	std::cout << "AVG RATIO:         " << res.ratio << std::endl;
	std::cout << "AVG COST:          " << res.cost << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);


	//cost1 = _G_COST - cost1;


	//res.L = -1;
	//res.K = m_;
	//res.c = c_;
	//res.time = mean_time * 1000;
	//res.recall = ((float)perform.NN_num) / (perform.num * k_);
	//res.ratio = ((float)perform.ratio) / (perform.res_num);
	//res.cost = ((float)0) / ((long long)perform.num);
	//res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}