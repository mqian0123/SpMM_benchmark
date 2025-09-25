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

## Dataset
Provided is the dataset used for evaluating performance across the different implementations:

Blocking: 
| Name                  | Rows        | Cols        | Nonzeros      | Kind                                           | Link                                                                 |
|-----------------------|------------:|------------:|--------------:|-----------------------------------------------|----------------------------------------------------------------------|
| road_usa              | 23,947,347 | 23,947,347 | 57,708,624    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/road_usa)                   |
| hugebubbles-00010     | 19,458,087 | 19,458,087 | 58,359,528    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/hugebubbles-00010)          |
| asia_osm              | 11,950,757 | 11,950,757 | 25,423,206    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/asia_osm)                   |
| 333SP                 | 3,712,815  | 3,712,815  | 22,217,266    | Undirected Graph                               | [link](https://sparse.tamu.edu/DIMACS10/333SP)                      |
| Spielman_k300         | 9,045,202  | 9,045,202  | 27,226,204    | Undirected Weighted Graph                      | [link](https://sparse.tamu.edu/FlowIPM22/Spielman_k300)             |

Scale-free:
| Name                  | Rows        | Cols        | Nonzeros      | Kind                                           | Link                                                                 |
|-----------------------|------------:|------------:|--------------:|-----------------------------------------------|----------------------------------------------------------------------|
| com-Orkut             | 3,072,441  | 3,072,441  | 234,370,166   | Undirected Graph With Communities (Social Network) | [link](https://sparse.tamu.edu/SNAP/com-Orkut)                   |
| twitter7              | 41,652,230 | 41,652,230 | 1,468,365,182 | Directed Graph (Social Network)                | [link](https://sparse.tamu.edu/SNAP/twitter7)                        |
| com-LiveJournal       | 3,997,962  | 3,997,962  | 69,362,378    | Undirected Graph With Communities (Social Network) | [link](https://sparse.tamu.edu/SNAP/com-LiveJournal)           |
| GAP-web               | 50,636,151 | 50,636,151 | 1,930,292,948 | Directed Weighted Graph (Web)                  | [link](https://sparse.tamu.edu/GAP/GAP-web)                         |
| uk-2002               | 18,520,486 | 18,520,486 | 298,113,762   | Directed Graph (Web)                           | [link](https://sparse.tamu.edu/LAW/uk-2002)                         |
| indochina-2004        | 7,414,866  | 7,414,866  | 194,109,311   | Directed Graph (Web)                           | [link](https://sparse.tamu.edu/LAW/indochina-2004)                  |

Diagonal:
| Name                  | Rows        | Cols        | Nonzeros      | Kind                                           | Link                                                                 |
|-----------------------|------------:|------------:|--------------:|-----------------------------------------------|----------------------------------------------------------------------|
| stokes                | 11,449,533 | 11,449,533 | 349,321,980   | Semiconductor Process Problem                  | [link](https://sparse.tamu.edu/VLSI/stokes)                          |
| Queen_4147            | 4,147,110  | 4,147,110  | 316,548,962   | 2D/3D Problem                                  | [link](https://sparse.tamu.edu/Janna/Queen_4147)                     |
| rajat31               | 4,690,002  | 4,690,002  | 20,316,253    | Circuit Simulation Problem                      | [link](https://sparse.tamu.edu/Rajat/rajat31)                        |
