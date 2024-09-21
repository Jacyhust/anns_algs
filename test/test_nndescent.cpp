#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_AVX512
#define EIGEN_DONT_PARALLELIZE


#include <sys/time.h>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include "assert.h"
// #include "io.h"
#include "../includes/kgraph.h"
// #include "/home/xizhao/ANNS/SIGMOD-Programming-Contest-2023/nn-descent/kgraph.h"
#include <omp.h>
// for avx
#include <x86intrin.h>

// for timer
#include <boost/timer/timer.hpp>
//#include "include/efanna2e/index_kdtree.h"
//#define timer timer_for_boost_progress_t



// for Eigen
#include <Eigen/Dense>
using namespace boost;

#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#endif
using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace kgraph;

#define _INT_MAX 2147483640





// Modify for Eigen
typedef Eigen::MatrixXf MyType;

const int DIM = 100;
const int K = 100;


typedef float ResultType;

void convertFloatPointerToMatrix(float** array, int rows, int cols, Eigen::MatrixXf& matrix) {
    matrix=Eigen::MatrixXf(cols,rows);

    // Step 2: Copy data from float** to the contiguous array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          matrix(j, i) = array[i][j];
        }
    }

}

// 【Eigen dist】
float compare(const MyType& a, const MyType& b) {
//  return ((a - b)*(a - b).transpose())(0,0);
  return ((a - b).transpose() *(a-b))(0,0);
}

// 【Eigen dist id】
float compare_with_id(const MyType& a, const MyType& b, uint32_t id_a, uint32_t id_b) {
//  return (urn ((a - b)*(a - b).transpose())(0,0);
//  return ((a - b).transpose() *(a-b))(0,0);
//  Eigen::MatrixXf tmp = (- 2 * (a.transpose()) * b);
//  cout<<KGraph::square_sums.size() <<" "<<KGraph::square_sums.rows() <<" "<<KGraph::square_sums.cols()<<"\n";
//  float ret = KGraph::square_sums(id_a, 0);
//  float ret = KGraph::square_sums(id_a, 0) + KGraph::square_sums(id_b, 0) -2 * (a.transpose()) * b)(0,0);
//  return ret;
  return (KGraph::square_sums(id_a, Eigen::all) + KGraph::square_sums(id_b, Eigen::all) + (((-2 * a.transpose()) * b)))(0,0);
}



// 【Eigen version】
typedef kgraph::VectorOracle<MyType, MyType> MyOracle;

#include "sol.h"
#include "../includes/alg.h"


void searchInKnng(const std::vector<std::vector<uint32_t>> &nngraph, Data& data, queryN* q, int start, int ef) {
			int cost = 0;
      //std::cout<<"size of knng: "<<nngraph.size()<<std::endl;
			lsh::timer timer;
			std::priority_queue<Res> accessed_candidates, top_candidates;
      int n=nngraph.size();
      std::vector<bool> visited(n,false);
			visited[start] = true;
			float dist = cal_L2sqr(q->queryPoint, data[start], data.dim);
			cost++;
			accessed_candidates.emplace(start, -dist);
			top_candidates.emplace(start, dist);

			while (!accessed_candidates.empty()) {
				Res top = accessed_candidates.top();
				if (-top.dist > top_candidates.top().dist) break;
				accessed_candidates.pop();

				for (auto& u : nngraph[top.id]) {
					if (visited[u]) continue;
					visited[u] = true;
					dist = cal_L2sqr(q->queryPoint, data[u], data.dim);
					cost++;
					accessed_candidates.emplace(u, -dist);
					top_candidates.emplace(u, dist);
					if (top_candidates.size() > ef) top_candidates.pop();
				}
			}

			while (top_candidates.size() > q->k) top_candidates.pop();

			q->res.resize(q->k);
			int pos = q->k;
			while (!top_candidates.empty()) {
				q->res[--pos] = top_candidates.top();
				top_candidates.pop();
			}
			q->time_total = timer.elapsed();
      q->cost=cost;
		}



int main(int argc, char **argv) {
  boost::timer::cpu_timer timer;
  string source_path = "dummy-data.bin";
//  string source_path = "contest-data-release-1m.bin";
//  string source_path = "contest-data-release-10m.bin";


  // Also accept other path for source data
  if (argc > 1) {
    source_path = string(argv[1]);
  }
  omp_set_num_threads(32);

  // Read data points
//  ReadBinEigen(source_path, KGraph::nodes);   // Eigen version
//   ReadBinEigenColMajor(source_path, KGraph::nodes);   // Eigen version

  std::string dataset = "mnist";
	if (argc > 1) {
		dataset = argv[1];
	}
	std::string argvStr[4];
	argvStr[1] = (dataset + ".data");
	argvStr[2] = (dataset + ".index");
	argvStr[3] = (dataset + ".bench_graph");

  std::cout << "Using FARGO for " << argvStr[1] << std::endl;
  Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));

  //KGraph::nodes=Eigen::Map<Eigen::MatrixXf>(prep.data.val, prep.data.N, prep.data.dim);
  convertFloatPointerToMatrix(prep.data.val, prep.data.N, prep.data.dim,KGraph::nodes);

  cout<<KGraph::nodes.cols()<<"\n";
  int n =  KGraph::nodes.cols();  // note: this should be rows rather than size!


  // K-graph related
  MyOracle oracle(KGraph::nodes, compare, compare_with_id);


  KGraph *index = KGraph::create();

  KGraph::IndexParams params;

  params.S = 100;
  params.K = 100;
  params.L=  265;
  params.R = 350;
  params.iterations= 8;


  params.recall = 1.0;
  params.delta = 0.0002;

  // 【For submit】
//  params.if_eval = false;
//  params.controls = 0;

  // 【For local evaluation】
  params.if_eval = true;
  params.controls= 100;


  index->build(oracle, params);

  printf("Build finished!\n");
  auto times = timer.elapsed();
  std::cerr << "Build time: " << times.wall / 1e9 <<"\n";


  auto times_get_knng = timer.elapsed();
  std::cerr << "Get KNNG time: " << (times_get_knng.wall - times.wall) / 1e9 <<"\n";

  size_t num_e=0;
  for(auto& edgeset:index->knng){
    num_e+=edgeset.size();
  }
  std::cout<<"Avg. Deg. = "<<(float) num_e/(prep.data.N)<<std::endl;
  // Save to ouput.bin
  //SaveKNNG(index->knng);
  auto times_save = timer.elapsed();
  std::cerr << "Save time: " << (times_save.wall - times_get_knng.wall) / 1e9 << "\n";

  float c_=0.5;
  int k_=50;
  int recall=0;
  float ratio=0.0f;
  auto& nngraph=index->knng;
  auto times1 = timer.elapsed();
  lsh::timer timer11;
  int cost=0;

  queryN** qs=new queryN*[prep.queries.N];
  std::cout<<"nq= "<<prep.queries.N<<std::endl;
  for(int i=0;i<prep.queries.N;++i){
    qs[i]=new queryN(0 , c_, k_, prep.queries[i],prep.queries.dim, 1.0f);
  }

  timer11.restart();
  for(int i=0;i<prep.queries.N;++i){
    //queryN q(0 , c_, k_, prep.queries[i],prep.queries.dim, 1.0f);
    searchInKnng(nngraph, prep.data, qs[i], 0, k_+100);
  }
  std::cout<<"Query Time= "<<(float) (timer11.elapsed()*1000)/(prep.queries.N)
  <<" ms."<<std::endl;


  for(int i=0;i<prep.queries.N;++i){
    cost+=qs[i]->cost;
    for(int k=0;k<k_;++k){
      ratio+=sqrt(qs[i]->res[k].dist)/prep.benchmark.innerproduct[i][k];
      //ratio+=(q.res[k].dist)/prep.benchmark.indice[i][k];
      for(int l=0;l<k_;++l){
          if(qs[i]->res[k].id==prep.benchmark.indice[i][l]){
              recall++;
              break;
          }
      }
    }
  }


  auto times2 = timer.elapsed();
  std::cout<<"Recall= "<<(float) recall/(prep.queries.N*k_)<<std::endl;
  std::cout<<"Ratio = "<<(float) ratio/(prep.queries.N*k_)<<std::endl;
  std::cout<<"Cost  = "<<(float) cost/(prep.queries.N)<<std::endl;
  std::cout<<"Query Time= "<<(float) (times2.wall-times1.wall)/(1e6*prep.queries.N)<<" ms."<<std::endl;
  
  return 0;
}
