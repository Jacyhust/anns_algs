SRCS=./test/test.cpp
OBJDIR=./build
RESDIR=./results
OBJS = $(patsubst ./test/test.cpp,$(OBJDIR)/test.o,$(SRCS))
TARGET = fg
CXX := g++
MKLROOT = /usr/include/mkl
OPTION = -I./ -DIN_PARALLEL  -I /usr/include/eigen3 -fopenmp -march=native -ffast-math -flto -I$(MKLROOT) -DNDEBUG
LFLAGS = -std=c++11 -O3 $(OPTION)  -L$(MKLROOT)/intel64 -lboost_timer -lmkl_intel_lp64 -lmkl_core  -lmkl_gnu_thread -lpthread -lm -ldl
CXXFLAGS := -std=c++17 -mavx512f -Ofast -lrt -DNDEBUG  -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0
# CXXFLAGS := -O3 -I /usr/include/eigen3 -fopenmp -mcmodel=medium -std=c++17 -mcpu=native #-fpic -mavx512f -lrt -DHAVE_CXX0X -ftree-vectorize -ftree-vectorizer-verbose=0 -openmp -DNDEBUG 

.PHONY:rnnd
.PHONY:srp
.PHONY:maria

all: $(TARGET) 

tb:./test/test_recall.cpp
	$(CXX) $(CXXFLAGS)  -o tb ./test/test_recall.cpp

$(OBJDIR)/%.o:./test/test.cpp
	@test -d $(OBJDIR) | mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET):$(OBJS)
	@test -d $(RESDIR) | mkdir -p $(RESDIR)
	@test -d indexes | mkdir -p indexes
	$(CXX) $(CXXFLAGS)  -o $@ $^

nndescent:./test/test_nndescent.cpp ./includes/kgraph.cpp
	@test -d $(RESDIR) | mkdir -p $(RESDIR)
	@if [ -e nnd ]; then rm nnd; fi
	$(CXX) $(LFLAGS) ./test/test_nndescent.cpp ./includes/kgraph.cpp -o nnd -lboost_timer -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

blas:./test/test_openblas.cpp
	$(CXX) $(CXXFLAGS) -o blas ./test/test_openblas.cpp -lopenblas

rnnd:./test/test_rnnd.cpp ./includes/RNNDescent.cpp
	@if [ -e rnnd ]; then rm rnnd; fi
	$(CXX) $(CXXFLAGS) -o rnnd ./test/test_rnnd.cpp ./includes/RNNDescent.cpp

srp:./test/test_srp.cpp ./includes/RNNDescent.cpp
	@if [ -e srp ]; then rm srp; fi
	$(CXX) $(CXXFLAGS) -o srp ./test/test_srp.cpp ./includes/RNNDescent.cpp -lopenblas

maria:./test/maria.cpp ./includes/RNNDescent.cpp
	$(CXX) $(CXXFLAGS) -o maria ./test/maria.cpp ./includes/RNNDescent.cpp -lopenblas


clean:
	rm -rf $(TARGET) $(OBJDIR) tb srp nnd rnnd maria
