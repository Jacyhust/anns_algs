/**
 * @file basis.h
 *
 * @brief A set of basic tools.
 */
#pragma once
#include <string>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <unordered_set>
#include <mutex>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string.h>
#include "patch_ubuntu.h"
#include "StructType.h"

 //#define COUNT_CC
//#define COUNT_PD

struct Res//the result of knns
{
	//dist can be:
	//1. L2-distance
	//2. The opposite number of inner product
	float dist = 0.0f;
	int id = -1;
	Res() = default;
	Res(int id_, float inp_) :id(id_), dist(inp_) {}
	Res(float inp_, int id_) :id(id_), dist(inp_) {}
	bool operator < (const Res& rhs) const {
		return dist < rhs.dist
			|| (dist == rhs.dist && id < rhs.id)
			;
	}

	constexpr bool operator > (const Res& rhs) const noexcept {
		return dist > rhs.dist;
	}

	constexpr bool operator == (const Res& rhs) const noexcept {
		return id == rhs.id;
	}
};

inline bool compareId(const Res& a, const Res& b) {
	return a.id == b.id;
}

namespace lsh
{
	class progress_display
	{
		public:
		explicit progress_display(
			unsigned long expected_count,
			std::ostream& os = std::cout,
			const std::string& s1 = "\n",
			const std::string& s2 = "",
			const std::string& s3 = "")
			: m_os(os), m_s1(s1), m_s2(s2), m_s3(s3)
		{
			restart(expected_count);
		}
		void restart(unsigned long expected_count)
		{
			//_count = _next_tic_count = _tic = 0;
			_expected_count = expected_count;
			m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
				<< m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
				<< std::endl
				<< m_s3;
			if (!_expected_count)
			{
				_expected_count = 1;
			}
		}
		unsigned long operator += (unsigned long increment)
		{
			std::unique_lock<std::mutex> lock(mtx);
			if ((_count += increment) >= _next_tic_count)
			{
				display_tic();
			}
			return _count;
		}
		unsigned long  operator ++ ()
		{
			return operator += (1);
		}

		//unsigned long  operator + (int x)
		//{
		//	return operator += (x);
		//}

		unsigned long count() const
		{
			return _count;
		}
		unsigned long expected_count() const
		{
			return _expected_count;
		}
		private:
		std::ostream& m_os;
		const std::string m_s1;
		const std::string m_s2;
		const std::string m_s3;
		std::mutex mtx;
		std::atomic<size_t> _count{ 0 }, _expected_count{ 0 }, _next_tic_count{ 0 };
		std::atomic<unsigned> _tic{ 0 };
		void display_tic()
		{
			unsigned tics_needed = unsigned((double(_count) / _expected_count) * 50.0);
			do
			{
				m_os << '*' << std::flush;
			} while (++_tic < tics_needed);
			_next_tic_count = unsigned((_tic / 50.0) * _expected_count);
			if (_count == _expected_count)
			{
				if (_tic < 51) m_os << '*';
				m_os << std::endl;
			}
		}
	};

	/**
	 * A timer object measures elapsed time, and it is very similar to boost::timer.
	 */
	class timer
	{
		public:
		timer() : time_begin(std::chrono::steady_clock::now()) {};
		~timer() {};
		/**
		 * Restart the timer.
		 */
		void restart()
		{
			time_begin = std::chrono::steady_clock::now();
		}
		/**
		 * Measures elapsed time.
		 *
		 * @return The elapsed time
		 */
		double elapsed()
		{
			std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
			return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) * 1e-6;// / CLOCKS_PER_SEC;
		}
		private:
		std::chrono::steady_clock::time_point time_begin;
	};
}

#include "fastL2_ip.h"
#include "distances_simd_avx512.h"
#include "patch_ubuntu.h"
//extern std::atomic<size_t> _G_COST;

// struct comp_cost{
// 	static std::atomic<size_t> _G_COST;
// };

inline float cal_inner_product(float* v1, float* v2, int dim)
{
	//++_G_COST;
#ifdef __AVX2__
	// printf("here!\n");
	// exit(-1);
	return faiss::fvec_inner_product_avx512(v1, v2, dim);
#else
	return calIp_fast(v1, v2, dim);

	float res = 0.0;
	for (int i = 0; i < dim; ++i) {
		res += v1[i] * v2[i];
	}
	return res;

	return calIp_fast(v1, v2, dim);
#endif

}

inline float cal_L2sqr(float* v1, float* v2, int dim)
{
	//++_G_COST;
#ifdef __AVX2__
	// std::cout<<"Support __AVX2__";
	// printf("here!\n");
	return (faiss::fvec_L2sqr_avx512(v1, v2, dim));
#else
	//printf("here11!\n");
	return calL2Sqr_fast(v1, v2, dim);
	float res = 0.0;
	for (int i = 0; i < dim; ++i) {
		res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return res;
#endif

}

inline float cal_inner_product(const float* v1, const float* v2, int dim)
{
	//++_G_COST;
#ifdef __AVX2__
	// printf("here!\n");
	// exit(-1);
	return faiss::fvec_inner_product_avx512(v1, v2, dim);
#else
	return calIp_fast(const_cast<float*>(v1), const_cast<float*>(v2), dim);

	float res = 0.0;
	for (int i = 0; i < dim; ++i) {
		res += v1[i] * v2[i];
	}
	return res;
#endif

}

inline float cal_L2sqr(const float* v1, const float* v2, int dim)
{
	//++_G_COST;
#ifdef __AVX2__
	// std::cout<<"Support __AVX2__";
	// printf("here!\n");
	return (faiss::fvec_L2sqr_avx512(v1, v2, dim));
#else
	//printf("here11!\n");
	return calL2Sqr_fast(const_cast<float*>(v1), const_cast<float*>(v2), dim);
	float res = 0.0;
	for (int i = 0; i < dim; ++i) {
		res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return res;
#endif

}


template <class T>
void clear_2d_array(T** array, int n)
{
	for (int i = 0; i < n; ++i) {
		delete[] array[i];
	}
	delete[] array;
}

inline float calInnerProductReverse(float* v1, float* v2, int dim) {
	return -cal_inner_product(v1, v2, dim);
}

#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
	if (numThreads <= 0) {
		numThreads = std::thread::hardware_concurrency();
	}

	if (numThreads == 1) {
		for (size_t id = start; id < end; id++) {
			fn(id, 0);
		}
	}
	else {
		std::vector<std::thread> threads;
		std::atomic<size_t> current(start);

		// keep track of exceptions in threads
		// https://stackoverflow.com/a/32428427/1713196
		std::exception_ptr lastException = nullptr;
		std::mutex lastExceptMutex;

		for (size_t threadId = 0; threadId < numThreads; ++threadId) {
			threads.push_back(std::thread([&, threadId] {
				while (true) {
					size_t id = current.fetch_add(1);

					if (id >= end) {
						break;
					}

					try {
						fn(id, threadId);
					}
					catch (...) {
						std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
						lastException = std::current_exception();
						/*
						 * This will work even when current is the largest value that
						 * size_t can fit, because fetch_add returns the previous value
						 * before the increment (what will result in overflow
						 * and produce 0 instead of current + 1).
						 */
						current = end;
						break;
					}
				}
				}));
		}
		for (auto& thread : threads) {
			thread.join();
		}
		if (lastException) {
			std::rethrow_exception(lastException);
		}
	}
}

#include <mutex>
#include <deque>
#include <set>

namespace threadPoollib
{
	typedef unsigned short int vl_type;

	class VisitedList {
		public:
		vl_type curV;
		//vl_type* mass;
		std::unordered_set<int> mass;
		unsigned int numelements;

		VisitedList(int numelements1) {
			curV = -1;
			numelements = numelements1;
			//mass = new vl_type[numelements];
		}

		void reset() {
			curV++;
			if (curV == 0) {
				//memset(mass, 0, sizeof(vl_type) * numelements);
				curV++;
			}
		};

		~VisitedList() {
			//delete[] mass; 
		}
	};
	///////////////////////////////////////////////////////////
	//
	// Class for multi-threaded pool-management of VisitedLists
	//
	/////////////////////////////////////////////////////////


	class VisitedListPool {
		std::deque<VisitedList*> pool;
		std::mutex poolguard;
		int numelements;

		public:
		VisitedListPool(int initmaxpools, int numelements1) {
			numelements = numelements1;
			for (int i = 0; i < initmaxpools; i++)
				pool.push_front(new VisitedList(numelements));
		}

		VisitedList* getFreeVisitedList() {
			VisitedList* rez;
			{
				std::unique_lock <std::mutex> lock(poolguard);
				if (pool.size() > 0) {
					rez = pool.front();
					pool.pop_front();
				}
				else {
					rez = new VisitedList(numelements);
				}
			}
			rez->reset();
			return rez;
		};

		void releaseVisitedList(VisitedList* vl) {
			std::unique_lock <std::mutex> lock(poolguard);
			pool.push_front(vl);
		};

		~VisitedListPool() {
			while (pool.size()) {
				VisitedList* rez = pool.front();
				pool.pop_front();
				delete rez;
			}
		};
	};
}

#if defined(unix) || defined(__unix__)
inline void localtime_s(tm* result, time_t* time) {
	if (localtime_r(time, result) == nullptr) {
		std::cerr << "Error converting time." << std::endl;
		std::memset(result, 0, sizeof(struct tm));
	}
}
#endif

inline void saveAndShow(float c, int k, std::string& dataset, std::vector<resOutput>& res)
{
	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);
	std::string query_result = ("results/Running_result.txt");
	std::ofstream os(query_result, std::ios_base::app);
	os.seekp(0, std::ios_base::end); // 

	time_t zero_point = 1635153971 - 17 * 3600 - 27 * 60;//Let me set the time at 2021.10.25. 17:27 as the zero point
	size_t diff = (size_t)(now - zero_point);
	//#if defined(unix) || defined(__unix__)
	//	llt lt(diff);
	//#endif
	//
	//	double date = ((float)(now - zero_point)) / 86400;
	//	float hour = date - floor(date);
	//	hour *= 24;
	//	float minute = hour = date - floor(date);


	std::stringstream ss;

	ss << "*******************************************************************************************************\n"
		<< "The result of Algs for MIPS on " << dataset << " is as follow: c=" << c << ", k=" << k
		<< "\n"
		<< "*******************************************************************************************************\n";

	ss << std::setw(12) << "algName"
		//<< std::setw(12) << "c"
		//<< std::setw(12) << "L"
		//<< std::setw(12) << "K"
		<< std::setw(12) << "Time"
		<< std::setw(12) << "Recall"
		<< std::setw(12) << "Ratio"
		<< std::setw(12) << "Cost"
		<< std::endl
		<< std::endl;
	for (int i = 0; i < res.size(); ++i) {
		ss << std::setw(12) << res[i].algName
			//<< std::setw(12) << res[i].c
			//<< std::setw(12) << res[i].L
			//<< std::setw(12) << res[i].K
			<< std::setw(12) << res[i].time
			<< std::setw(12) << res[i].recall
			<< std::setw(12) << res[i].ratio
			<< std::setw(12) << res[i].cost
			<< std::endl;
	}
	ss << "\n******************************************************************************************************\n"
		<< "                                                                                    "
		<< ltm->tm_mon + 1 << '-' << ltm->tm_mday << ' ' << ltm->tm_hour << ':' << ltm->tm_min
		<< "\n*****************************************************************************************************\n\n\n";
	std::cout << ss.str();
	os << ss.str();
	os.close();  delete[]ltm;
	//delete[] ltm;
}

template <class T>
inline bool myFind(T* begin, T* end, const T& val)
{
	for (T* iter = begin; iter != end; ++iter) {
		if (*iter == val) return true;
	}
	return false;
}

inline int isUnique(std::vector<Res>& vec) {
	int len = vec.size();
	std::set<int> s;
	for (auto& x : vec) {
		s.insert(x.id);
	}
	//std::set<T> s(vec.begin(), vec.end());
	return len == s.size();
}

inline int isUnique(std::vector<int>& vec) {
	int len = vec.size();

	std::set<int> s(vec.begin(), vec.end());
	return len == s.size();
}

inline int isUnique(Res* sta, Res* end) {
	int len = end - sta;
	std::set<int> s;
	for (auto u = sta; u < end; ++u) {
		s.insert(u->id);
	}
	return len == s.size();
}

template <class T, class U>
int isUnique(std::map<U, T>& vec) {
	int len = 0;
	std::set<T> s;
	for (auto& x : vec) {
		s.insert(x.second);
		++len;
	}
	return len == s.size();
}


#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#define NOMINMAX

#undef max

#undef min
#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>


#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
inline size_t getCurrentRSS() {
#if defined(_WIN32)
	/* Windows -------------------------------------------------- */
	PROCESS_MEMORY_COUNTERS info;
	GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
	return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
	/* OSX ------------------------------------------------------ */
	struct mach_task_basic_info info;
	mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
		(task_info_t)&info, &infoCount) != KERN_SUCCESS)
		return (size_t)0L;      /* Can't access? */
	return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
	/* Linux ---------------------------------------------------- */
	long rss = 0L;
	FILE* fp = NULL;
	if ((fp = fopen("/proc/self/statm", "r")) == NULL)
		return (size_t)0L;      /* Can't open? */
	if (fscanf(fp, "%*s%ld", &rss) != 1) {
		fclose(fp);
		return (size_t)0L;      /* Can't read? */
	}
	fclose(fp);
	return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
	/* AIX, BSD, Solaris, and Unknown OS ------------------------ */
	return (size_t)0L;          /* Unsupported. */
#endif
}

inline bool exists_test(const std::string& name) {
	//return false;
	std::ifstream f(name.c_str());
	return f.good();
}

#if defined _MSC_VER
#include <intrin.h>
#endif

//For two uint64_t numbers k1 and k2, compute the number of their differences in terms of bits
inline int bitCounts(uint64_t k1, uint64_t k2)
{
#if defined(__GNUC__)
	return __builtin_popcount(k1 ^ k2);
#elif defined _MSC_VER
	return (int)__popcnt(k1 ^ k2);
#else
	std::cout << BOLDRED << "WARNING:" << RED << "bitCounts Undefined in this compipler.\nYou can find the related functions in your system. \n" << RESET;
	exit(-1);
#endif

}

//For two uint64_t numbers k1 and k2, compute the number of their differences in terms of bits
inline int bitCounts(uint64_t* k1, uint64_t* k2)
{
#if defined(__GNUC__)
	return __builtin_popcount((*k1) ^ (*k2));
#elif defined _MSC_VER
	return (int)__popcnt((*k1) ^ (*k2));
#else
	std::cout << BOLDRED << "WARNING:" << RED << "bitCounts Undefined in this compipler.\nYou can find the related functions in your system. \n" << RESET;
	exit(-1);
#endif

}

#if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L) && (_MSC_VER >= 1913))
#include <shared_mutex>
typedef std::shared_mutex mp_mutex;
//In C++17 format, read_lock can be shared
typedef std::shared_lock<std::shared_mutex> read_lock;
typedef std::unique_lock<std::shared_mutex> write_lock;
#else
typedef std::mutex mp_mutex;
//Not in C++17 format, read_lock is the same as write_lock and can not be shared
typedef std::unique_lock<std::mutex> read_lock;
typedef std::unique_lock<std::mutex> write_lock;
#endif // _HAS_CXX17

namespace lsh {
	//My defined priority queue for Res 
	//Compared to std::priority_queue:
	//1.Support duplicating
	//2.Support for accessing the array
	//template <typename T>
	struct  priority_queue
	{
		private:
		int size_ = 0;
		int capacity = 0;
		Res* data_ = nullptr;

		public:
		priority_queue() = default;
		priority_queue(int K) {
			reset(K);
		}

		void reset(int K) {
			capacity = K + 1;
			delete[] data_;
			data_ = new Res[capacity];
		}

		void emplace(int id, float dist) {
			data_[size_++] = Res(id, dist);
			std::push_heap(data_, data_ + size_);
		}

		void push(Res res) {
			data_[size_++] = res;
			std::push_heap(data_, data_ + size_);
		}

		void emplace_with_duplication(int id, float dist) {
			data_[size_] = Res(id, dist);
			for (int i = 0;i < size_;++i) {
				if (compareId(data_[size_], data_[i])) return;
			}
			size_++;
			std::push_heap(data_, data_ + size_);
		}

		void pop() {
			std::pop_heap(data_, data_ + size_);
			size_--;
		}

		int size() {
			return size_;
		}

		bool empty() {
			return size_ == 0;
		}

		Res& top() {
			return data_[0];
		}

		Res*& data() {
			return data_;
		}

		void clear() {
			delete[] data_;
			data_ = nullptr;
		}

		~priority_queue() {
			if (data_) delete[] data_;
		}
	};

}