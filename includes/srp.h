#pragma once
#include "utils/StructType.h"
#include "utils/Preprocess.h"
#include <cmath>
#include <assert.h>
#include <vector>
#include <queue>
#include <cfloat>

#define USE_BLAS

#ifdef USE_BLAS
#include <cblas.h>
#endif

namespace lsh
{
	struct srpPair{
		uint32_t val;
		int id;

		srpPair(int id_, uint32_t hashval):id(id_),val(hashval){}

		bool operator < (const srpPair& rhs) const { return val < rhs.val;}
	};

	//My implement for a simple sign random prejection LSH function class
	class srp{
		//int N=0;
		
		std::vector<std::vector<uint32_t>> hashvals;
		std::vector<std::vector<srpPair>> hash_tables;
		float* rndAs=nullptr;
		int dim=0;
		// Number of hash functions
		int S=0;
		//#L Tables; 
		int L=0;
		// Dimension of the hash table
		int K=0;

	public:

		srp()=default;

		srp(Data& data, std::vector<std::vector<int>>& part_map, int N_,int dim_,int L_,int K_){
			//N=N_;
			dim=dim_;
			L=L_;
			K=K_;
			S=L*K;
			hashvals.resize(N_);

			std::cout << std::endl << "START HASHING..." << std::endl << std::endl;
			lsh::timer timer;

			std::cout << "SETTING HASH PARAMETER..." << std::endl;
			timer.restart();
			SetHash();
			std::cout << "SETTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

			std::cout << "COMPUTING HASH..." << std::endl;
			timer.restart();
			GetHash(data);
			std::cout << "COMPUTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

			std::cout << "BUILDING INDEX..." << std::endl;
			std::cout << "THERE ARE " << L << " " << K << "-D HASH TABLES." << std::endl;
			timer.restart();

			if(part_map.empty()) GetTables();
			else GetTables(part_map);

			std::cout << "BUILDING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
		}

		void SetHash(){
			rndAs = new float [S*dim];
			//hashpar.rndAs2 = new float* [S];
			
			//std::mt19937 rng(int(std::time(0)));
			std::mt19937 rng(int(0));
			std::normal_distribution<float> nd;
			for(int i=0;i<S*dim;++i) rndAs[i]=(nd(rng));
		}


		void GetHash(Data& data){
#ifdef USE_BLAS
			int m=hashvals.size();
			int k=dim;
			int n=S;

			float* A=data.base;
			float* B=rndAs;
			float* C=new float[m*n];

			memset(C,0.0f,m*n*sizeof(float));
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
						m, n, k, 1.0, A, k, B, k, 0.0, C, n);

			
			for(int i=0;i<hashvals.size();++i){
				hashvals[i].resize(L,0);
				for(int j=0;j<L;++j){
					for(int l=0;l<K;++l){
						float val=C[i*S+j*K+l];
						//cal_inner_product(data[i],rndAs+(j*K+l)*dim,dim);
						if(val>0) hashvals[i][j]|=(1<<l);
					}
					
				}
			}
#else
			for(int i=0;i<hashvals.size();++i){
				hashvals[i].resize(L,0);
				for(int j=0;j<L;++j){
					for(int l=0;l<K;++l){
						float val=cal_inner_product(data[i],rndAs+(j*K+l)*dim,dim);
						if(val>0) hashvals[i][j]|=(1<<l);
					}
					
				}
			}
#endif

			// for(int i=0;i<10;++i){
			// 	for(int j=0;j<L;++j){
			// 		std::cout<<hashvals[i][j]<<" ";
					
			// 	}
			// 	std::cout<<std::endl;
			// }
		}

		void GetTables(std::vector<std::vector<int>>& part_map){
			int num_parti=part_map.size();
			hash_tables.resize(num_parti*L);
			for(int i=0;i<num_parti;++i){
				auto& part=part_map[i];
				for(auto& id:part){
					for(int j=0;j<L;++j){
						hash_tables[i*L+j].emplace_back(id,hashvals[id][j]);
					}
				}
			}

			for(auto& table:hash_tables){
				std::sort(table.begin(),table.end());
			}
		}

		void GetTables(){
			hash_tables.resize(L);
			for(int i=0;i<hashvals.size();++i){
				int id=i;
				for(int j=0;j<L;++j){
					hash_tables[j].emplace_back(id,hashvals[id][j]);
				}
			}

			for(auto& table:hash_tables){
				std::sort(table.begin(),table.end());
			}
		}
	};
}

