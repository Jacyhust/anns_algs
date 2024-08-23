#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "../includes/utils/StructType.h"
#include "../includes/utils/basis.h"
#include "../includes/hcnngLite.h"
#include "../includes/fastGraph.h"
using namespace std;

// class Solution {
//     hcnngLite::hcnng<calInnerProductReverse>* hcnng=nullptr;
// public:
//     void build(int d, const vector<float>& base){
//         int minsize_cl = 750;
//         int num_cl = 20;
//         int max_mst_degree = 3;

//         std::string dataset="test";
//         Data data;

//         data.N=base.size()/d;
//         data.dim=d;
//         //data.val=nullptr;
//         data.val=new float*[data.N];
//         //float* pos=&(base[0]);
//         //const float* pos=base.data();
//         float* pos=const_cast<float*>(base.data());
//         for(int i=0;i<data.N;++i){
//             //data.val[i]=new float[d];
//             data.val[i]=pos+i*d;
//         }

//         hcnng=new hcnngLite::hcnng<calInnerProductReverse>(dataset, data, "test.index_hcnng", "index_result.txt",
//             minsize_cl, num_cl, max_mst_degree, 1);
//     }

//     void search(const vector<float>& query,int* res){
//         int dim=hcnng->data.dim;
//         //int nq=query.size()/dim;
//         int k_=10;
//         int c_=1.0f;
//         float m_=1.0f;
//         float* pos=const_cast<float*>(query.data());

//         queryN q(0 , c_, k_, pos,dim, m_);
//         hcnng->knn(&q);

//         for (int j = 0; j < 10; j++) {
//             res[j]=q.res[j].id;
//         }
//     }
// };

class Solution {
    //hcnngLite::hcnng<calInnerProductReverse>* hcnng=nullptr;
    fastGraph* fsG=nullptr;
    Data data;
public:
    void build(int d, const vector<float>& base){
        float c = 1.5;
        unsigned k = 50;
        unsigned L = 8, K = 10;//NUS
        //L = 10, K = 5;
        float beta = 0.1;
        unsigned Qnum = 100;
        float W = 1.0f;
        int T = 24;
        int efC = 80;
        L = 2;
        K = 18;
        double pC = 0.95, pQ = 0.9;

        std::string dataset="test";
        

        data.N=base.size()/d;
        data.dim=d;
        //data.val=nullptr;
        data.val=new float*[data.N];
        //float* pos=&(base[0]);
        //const float* pos=base.data();
        float* pos=const_cast<float*>(base.data());
        for(int i=0;i<data.N;++i){
            //data.val[i]=new float[d];
            data.val[i]=pos+i*d;
        }

        Parameter param1(data, L, K, 1.0f);
	    param1.W = 0.3f;
        std::cout << "Build!\n";
        divGraph* divG = new divGraph(data, param1, "indexes/test.file", T, efC, pC, pQ);
        delete divG->visited_list_pool_;
        std::cout << "Loading FastGraph...\n";
	    fsG = new fastGraph(divG);
        delete divG;
    }

    void search(const vector<float>& query,int* res){
        int dim=fsG->dim;
        //int nq=query.size()/dim;
        int k_=10;
        int c_=1.0f;
        float m_=1.0f;
        float* pos=const_cast<float*>(query.data());

        queryN q(0 , c_, k_, pos,dim, m_);
        fsG->knn(&q);

        for (int j = 0; j < 10; j++) {
            res[j]=q.res[j].id;
        }
    }
};

// class Solution1 {
//     //hcnngLite::hcnng<calInnerProductReverse>* hcnng=nullptr;
//     fastGraph* fsG=nullptr;
// public:
//     void build(int d, const vector<float>& base){
        
//     }

//     void search(const vector<float>& query,int* res){
//         int dim=fsG->dim;
//         //int nq=query.size()/dim;
//         int k_=10;
//         int c_=1.0f;
//         float m_=1.0f;
//         float* pos=const_cast<float*>(query.data());

//         queryN q(0 , c_, k_, pos,dim, m_);
//         fsG->knn(&q);

//         for (int j = 0; j < 10; j++) {
//             res[j]=q.res[j].id;
//         }
//     }
// };