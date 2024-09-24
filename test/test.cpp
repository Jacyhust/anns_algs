#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "sol.h"
#include "../includes/alg.h"
//#include "mf_alsh.h"

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

//std::atomic<size_t> _G_COST=0;

int main(int argc, char const* argv[])
{
	std::string dataset = "gist";
	if (argc > 1) {
		dataset = argv[1];
	}
	std::string argvStr[4];
	argvStr[1] = (dataset + ".data");
	argvStr[2] = (dataset + ".index");
	argvStr[3] = (dataset + ".bench_graph");

	std::cout << "Using FARGO for " << argvStr[1] << std::endl;
	Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));
	
    int d=prep.data.dim;
    std::vector<float> points(d*prep.data.N);
    
    for(int i=0;i<prep.data.N;++i){
        for(int j=0;j<prep.data.dim;++j){
            points[i*d+j]=prep.data[i][j];
        }
    }
    std::cout<<"Begin solution!\n";
    Solution s;
    s.build(d, points);
    int recall=0;
    for(int i=0;i<100;++i){
        std::vector<float> q(prep.queries[i],prep.queries[i]+d);
        int res[10];
        s.search(q,res);

        for(int k=0;k<10;++k){
            for(int l=0;l<10;++l){
                if(res[k]==prep.benchmark.indice[i][l]){
                    recall++;
                    break;
                }
            }
        }
    }

    std::cout<<"Recall= "<<(float) recall/1000<<std::endl;
	//saveAndShow(c, k, dataset, res);
	return 0;
}