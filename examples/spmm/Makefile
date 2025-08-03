# source /opt/intel/oneapi/setvars.sh

# Compiler and flags
CXX       = icpx
CXXFLAGS  = -std=c++20 -O2 -g -w \
            -fno-inline -fno-inline-functions

n ?= 1
CXXFLAGS += -DRHSDIM=$(n)

TARGET    = spmm
SRC       = spmm.cpp

# Include paths
INCLUDES  = -I/home/mqian01/binsparse-reference-c/include \
            -I$(HOME)/intel/oneapi/advisor/2025.2/include \
            -I$(MKLROOT)/include \
            -I/usr/include/hdf5/serial \
            -I/usr/include/cjson \
			-I/home/mqian01/SparseRooflineBenchmark/examples/spmm/csbv3_website

# Library paths
LDFLAGS   = -L/home/mqian01/binsparse-reference-c/build \
            -Wl,-rpath,/home/mqian01/binsparse-reference-c/build \
            -L$(MKLROOT)/lib/intel64 \
            -L$(HOME)/intel/oneapi/advisor/2025.2/sdk/lib64

# Libraries
LIBS      = -lbinsparse-rc -lhdf5_serial -lcjson \
            -Wl,--start-group \
                -lmkl_intel_lp64 \
                -lmkl_sequential \
                -lmkl_core \
            -Wl,--end-group \
            -lpthread -lm -ldl \
            -littnotify

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)
