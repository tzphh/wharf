#include "libs/compressed_trees/pbbslib/parallel.h"
#include <iostream>
int main(int argc, char** argv) {
    size_t n = 10;

    // i [0, i)
    parallel_for(0, n, [&](size_t i) {
        std::cout << i << std::endl;
    });
    return 0;
}