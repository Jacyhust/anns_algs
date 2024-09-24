#include <iostream>
#include <cblas.h>
#include <string>
#include <omp.h>
#include "sol.h"
#include "../includes/alg.h"

using T=float;

bool isEqual(T* A, T* B, int size){
    //bool res=true;
    for(int i=0;i<size;++i){
        //if(A[i]!=B[i]) return false;
        if(fabs(A[i]-B[i])>0.0001){
            std::cout<<A[i]<<","<<B[i]<<std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    std::string dataset = "audio";
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

    int m=7712;
    int k=prep.data.dim;
    int n=1;
    int repetition=1;

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
    memset(gt,0.0f,m*n*sizeof(float));
    std::cout<<"Setting Time= "<<(timer.elapsed()*1000)<<" ms.\n"<<std::endl;

    timer.restart();
//#pragma omp parallel for
    for(int l=0;l<repetition;++l){
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                gt[i*n+j]=cal_inner_product(prep.data[i],prep.queries[j],k);
            }
        }
    }
    
    std::cout<<"VxV in-data Time= "<<(timer.elapsed()*1000)<<" ms.\n"<<std::endl;
    float time_gt=timer.elapsed()*1000;

    timer.restart();
    //float* a=new float[k];
    //float* b=new float[k];
//#pragma omp parallel for
    for(int l=0;l<repetition;++l){
        for(int i=0;i<m;++i){
            //memcpy(a,prep.data[i],k*sizeof(float));
            //float* a=A+i*k;
            for(int j=0;j<n;++j){
                //memcpy(b,prep.queries[j],k*sizeof(float));
                //float* b=B2+j*k;
                gt[i*n+j]=cal_inner_product(A+i*k,B2+j*k,k);

            }
        }
    }
    
    std::cout<<"VxV out-data Time= "<<(timer.elapsed()*1000)<<" ms.\n"<<std::endl;

    timer.restart();
//#pragma omp parallel for
    for(int l=0;l<repetition;++l){
        float* A=new float[m*k];
        float* B1=new float[k*n];
        C=new float[m*n];
        for(int i=0;i<m;i++){
            memcpy(A+i*k,prep.data[i],k*sizeof(float));
        }

        for(int i=0;i<n;i++){
            for(int j=0;j<k;++j){
                B1[j*n+i]=prep.queries[i][j];
            }
        }

        memset(C,0.0f,m*n*sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, k, B1, n, 0.0, C, n);

        
    }
    

    std::cout<<"NoTrans Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;
    std::cout<<"Speedup=      "<<time_gt/(timer.elapsed()*1000)<<std::endl;
    std::cout<<"IsCorrect?    "<<isEqual(gt,C,m*n)<<std::endl<<std::endl;

    //memset(C,0.0f,m*n*sizeof(float));
    timer.restart();
//#pragma omp parallel for
    for(int l=0;l<repetition;++l){
        float* A=new float[m*k];
        float* B1=new float[k*n];
        float* B2=new float[k*n];
        C=new float[m*n];
        for(int i=0;i<m;i++){
            memcpy(A+i*k,prep.data[i],k*sizeof(float));
        }

        for(int i=0;i<n;i++){
            for(int j=0;j<k;++j){
                B2[i*k+j]=prep.queries[i][j];
            }
        }

        memset(C,0.0f,m*n*sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, n, k, 1.0, A, k, B2, k, 0.0, C, n);
    }
    std::cout<<"Trans Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;
    std::cout<<"Speedup=      "<<time_gt/(timer.elapsed()*1000)<<std::endl;
    std::cout<<"IsCorrect?    "<<isEqual(gt,C,m*n)<<std::endl<<std::endl;


    if(n==1){
        timer.restart();
    //#pragma omp parallel for
        for(int l=0;l<repetition;++l){
            // float* A=new float[m*k];
            // float* B1=new float[k*n];
            // float* B2=new float[k*n];
            //C=new float[m*n];
            for(int i=0;i<m;i++){
                memcpy(A+i*k,prep.data[i],k*sizeof(float));
            }

            memcpy(B2,prep.queries[0],k*sizeof(float));

            memset(C,0.0f,m*n*sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                m, k, 1.0, A, k, B2, 1, 0.0, C, 1);

            if(!isEqual(gt,C,m*n)){
                std::cout<<"Error!\n";
            }
        }
        std::cout<<"M*V Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;
        std::cout<<"Speedup=      "<<time_gt/(timer.elapsed()*1000)<<std::endl;
        std::cout<<"IsCorrect?    "<<isEqual(gt,C,m*n)<<std::endl<<std::endl;
    }

    if(n==1){
        float* A2=new float[m*k];
        __builtin_prefetch(A2, 0, 1);
        timer.restart();
        int L=1;
        
    //#pragma omp parallel for
        for(int l=0;l<repetition;++l){
            // float* A=new float[m*k];
            // float* B1=new float[k*n];
            // float* B2=new float[k*n];
            //C=new float[m*n];
            for(int i=0;i<m;i++){
                //  if (i + L < m) {
                //     __builtin_prefetch(prep.data[i + L], 0, 1);
                // }
                // else{
                //     __builtin_prefetch(prep.queries[0], 0, 1);
                // }
                memcpy(A2+i*k,prep.data[i],k*sizeof(float));
            }
            //memmove(A2,A,m*k*sizeof(float));
            memcpy(B2,prep.queries[0],k*sizeof(float));

            memset(C,0.0f,m*n*sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                m, k, 1.0, A2, k, B2, 1, 0.0, C, 1);

            if(!isEqual(gt,C,m*n)){
                std::cout<<"Error!\n";
            }
        }
        std::cout<<"M*V+pf Time= "<<(timer.elapsed()*1000)<<" ms."<<std::endl;
        std::cout<<"Speedup=      "<<time_gt/(timer.elapsed()*1000)<<std::endl;
        std::cout<<"IsCorrect?    "<<isEqual(gt,C,m*n)<<std::endl<<std::endl;
    }
    return 0;
}
