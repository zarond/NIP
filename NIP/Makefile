CXX = g++
NVCC = nvcc

# флаги компиляции
CXXFLAGS =
CXXFLAGS += -std=c++11 -O2 -march=native
CXXFLAGS += $(shell pkg-config --cflags blitz)
CXXFLAGS += -fopenmp
CXXFLAGS += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
CXXFLAGS += -rdc=true

# флаги сборки (библиотеки)
LDFLAGS =
LDFLAGS += $(shell pkg-config --libs blitz)
LDFLAGS += -I..\common\inc

SOURCES = main.cu
BINARY = NIP

#VISUAL = visual
#VISUAL_SOURCES = main.cu
#VISUAL_LDFLAGS = $(shell pkg-config --libs glut) -lGL

$(BINARY): $(SOURCES) Makefile
$(BINARY): CustomTimer.cpp functionCPU.cpp functionGPU.cu helper_kernels.cu
#$(BINARY): CustomTimer.cpp functionCPU.cpp functionGPU.cu helper_kernels.cu CustomTimer.h functionCPU.h functionGPU.cuh helper_kernels.cuh stb_image.h types.h
$(BINARY):
	$(NVCC) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $(BINARY)

#$(VISUAL): Makefile $(VISUAL_SOURCES)
#$(VISUAL): autoreg_driver.hh autoreg.hh parallel_mt.hh sysv.hh types.hh voodoo.hh
#$(VISUAL):
#	$(CXX) $(CXXFLAGS) $(VISUAL_SOURCES) $(VISUAL_LDFLAGS) -o $(VISUAL)

#clean:
#	rm -f $(BINARY) $(VISUAL)
