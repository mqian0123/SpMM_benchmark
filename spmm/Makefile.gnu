# Compiler and flags
CXX       = g++
CXXFLAGS  = -std=c++20 -O3 -g -fopenmp \
            -march=native -funroll-loops -fstrict-aliasing \
            -DLIKWID_PERFMON


TARGET    = spmm
SRC       = spmm.cpp

# Include paths
INCLUDES  = -I$(MKLROOT)/include \
			-I$(LIKWID_ROOT)/include

# Library paths
LDFLAGS   = -L$(MKLROOT)/lib/intel64 \
			-L$(LIKWID_ROOT)/lib \
            -Wl,-rpath,$(LIKWID_ROOT)/lib

# Libraries
LIBS      = -Wl,--start-group \
                -lmkl_intel_lp64 \
                -lmkl_intel_thread \
                -lmkl_core \
                -liomp5 \
            -Wl,--end-group \
            -lpthread -lm -ldl \
			-llikwid -fopenmp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)


clean:
	rm -f $(TARGET)
