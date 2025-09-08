## Installation

The Benchmark utilizes Intel Advisor and Math Kernel Library (MKL). Please install Intel's OneAPI
Base Toolkit.

## Usage
Inside spmm.cpp, change the RHSDIM variable to be the number of columns in B you want (Ex. 16, 32, etc).

```bash
cd spmm
make
./spmm <matrix_file.mtx> <n_columns_B>
```

Optional: The benchmark has a Julia file to verify correctness. To install Julia, please follow
the instructions on the [Julia website](https://julialang.org/downloads/).

