CXX = g++
NVCC = nvcc

# \u0444\u043b\u0430\u0433\u0438 \u043a\u043e\u043c\u043f\u0438\u043b\u044f\u0446\u0438\u0438
CXXFLAGS =
CXXFLAGS += -std=c++11 -O2 
#-march=native
CXXFLAGS += $(shell pkg-config --cflags blitz)
CXXFLAGS += $(shell pkg-config --libs glut) -lGL
#CXXFLAGS += $(shell pkg-config --cflags glew)
CXXFLAGS += -Xcompiler -fopenmp
CXXFLAGS += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
CXXFLAGS += -rdc=true
CXXFLAGS += -I../common/inc/

# \u0444\u043b\u0430\u0433\u0438 \u0441\u0431\u043e\u0440\u043a\u0438 (\u0431\u0438\u0431\u043b\u0438\u043e\u0442\u0435\u043a\u0438)
LDFLAGS =
LDFLAGS += $(shell pkg-config --libs blitz)
LDFLAGS += -I../common/inc/

SOURCES = functionCPU.cpp functionGPU.cu helper_kernels.cu main.cu 
SOURCE1 = functionCPU.cpp
SOURCE2 = functionGPU.cu
SOURCE3 = helper_kernels.cu
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