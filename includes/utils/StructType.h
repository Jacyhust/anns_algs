#pragma once

struct Data
{
	// Dimension of data
	int dim;
	// Number of data
	int N;
	// Data matrix
	float** val;
	int offset;
	// No safety checking!!!
	float*& operator[](int i) { return val[i]; }

//private:
	float* base=nullptr;

	~Data(){
		delete [] base;
		//delete [] val;
	}
};

struct Ben
{
	int N;
	int num;
	int** indice;
	float** innerproduct;
};

struct HashParam
{
	// the value of a in S hash functions
	float** rndAs1=nullptr;
	// the value of a in S hash functions
	float** rndAs2=nullptr;

	float** rndAs = nullptr;
	float* rndBs = nullptr;
};

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

struct tPoints {
	int u;
	int v;
};