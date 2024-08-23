SRCS=./test/test.cpp
OBJDIR=./build
RESDIR=./results
OBJS = $(patsubst ./test/test.cpp,$(OBJDIR)/test.o,$(SRCS))
TARGET = fg
CXX := g++
# CXXFLAGS := -std=c++17 -mavx512f -Ofast -lrt -DNDEBUG  -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0
CXXFLAGS := -O3 -I /usr/include/eigen3 -fopenmp -mcmodel=medium -std=c++17 -mcpu=native #-fpic -mavx512f -lrt -DHAVE_CXX0X -ftree-vectorize -ftree-vectorizer-verbose=0 -openmp -DNDEBUG 

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

clean:
	rm -rf $(TARGET) $(OBJDIR) tb
