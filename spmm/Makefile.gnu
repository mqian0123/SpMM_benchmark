# source /opt/intel/oneapi/setvars.sh

# Compiler and flags
CXX       = g++
CXXFLAGS  = -std=c++20 -O3 -g -w -fopenmp \
            -march=native -funroll-loops -fstrict-aliasing


TARGET    = spmm
SRC       = spmm.cpp

# Include paths
# INCLUDES  = -I$(HOME)/intel/oneapi/advisor/2025.2/include \
#             -I$(MKLROOT)/include \

# Library paths
# LDFLAGS   = -L$(MKLROOT)/lib/intel64 \
#             -L$(HOME)/intel/oneapi/advisor/2025.2/sdk/lib64

# Libraries
# LIBS      = -Wl,--start-group \
#                 -lmkl_intel_lp64 \
#                 -lmkl_intel_thread \
#                 -lmkl_core \
#                 -liomp5 \
#             -Wl,--end-group \
#             -lpthread -lm -ldl \
#             -littnotify

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)


clean:
	rm -f $(TARGET)
