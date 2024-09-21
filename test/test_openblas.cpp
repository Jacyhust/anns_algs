#include <iostream>
#include <cblas.h>
#include <string>
#include "sol.h"
#include "../includes/alg.h"

using T=float;

bool isEqual(T* A, T* B, int size){
    //bool res=true;
    for(int i=0;i<size;++i){
        if(A[i]!=B[i]) return false;
    }
    return true;
}

int main(int argc, char **argv) {
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

    lsh::timer timer;

    int m=1920;
    int k=prep.data.dim;
    int n=200;

    float* A=new float[m*k];
    float* B1=new float[k*n];
    float* B2=new float[k*n];
    float* C=new float[m*n];
    float* gt=new float[m*n];
    for(int i=0;i<m;i++){
        memcpy(A+i*k,prep.data[i],k*sizeof(float));
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<k;++j){
            B2[i*k+j]=prep.queries[i][j];
            B1[j*n+i]=prep.queries[i][j];
        }
    }

    memset(C,0.0f,m*n*sizeof(float));

    std::cout<<"Setting Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;

    timer.restart();
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            gt[i*n+j]=cal_inner_product(prep.data[i],prep.queries[j],k);
        }
    }

    std::cout<<"VxV Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;

    timer.restart();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, k, B1, n, 0.0, C, n);

    std::cout<<"NoTrans Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;
    std::cout<<"IsCorrect?    "<<isEqual(gt,C,m*n)<<std::endl;

    memset(C,0.0f,m*n*sizeof(float));
    timer.restart();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0, A, k, B2, k, 0.0, C, n);

    std::cout<<"Trans Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;
    std::cout<<"IsCorrect?    "<<isEqual(gt,C,m*n)<<std::endl;
    return 0;
}
