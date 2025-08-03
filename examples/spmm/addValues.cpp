#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.mtx output.mtx\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Error: Cannot open input file " << argv[1] << "\n";
        return 1;
    }

    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "Error: Cannot open output file " << argv[2] << "\n";
        return 1;
    }

    std::string line;
    bool headerProcessed = false;

    while (std::getline(in, line)) {
        // Preserve comments
        if (line.size() > 0 && line[0] == '%') {
            // Fix the header to say "real" instead of "pattern"
            if (!headerProcessed && line.find("pattern") != std::string::npos) {
                size_t pos = line.find("pattern");
                if (pos != std::string::npos) {
                    line.replace(pos, 7, "real");
                }
            }
            if (!headerProcessed && line.find("symmetric") != std::string::npos) {
                size_t pos = line.find("symmetric");
                if (pos != std::string::npos) {
                    line.replace(pos, 9, "general");
                }
            }
            out << line << "\n";
            continue;
        }

        std::istringstream iss(line);

        if (!headerProcessed) {
            // First non-comment line is the matrix dimensions + nnz
            out << line << "\n";
            headerProcessed = true;
        } else {
            // Row and column from pattern file
            int row, col;
            if (!(iss >> row >> col)) continue; // skip malformed lines
            out << row << " " << col << " 1.0\n";
        }
    }

    std::cout << "Conversion complete: wrote " << argv[2] << "\n";
    return 0;
}
