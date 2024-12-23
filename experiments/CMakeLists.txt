# -------------------------- Wharf ----------------------------
# 添加头文件路径
include_directories(${CMAKE_SOURCE_DIR}/utils)

option(CLANG     OFF)
option(CILK      OFF)
option(OPENMP    OFF)
option(HOMEGROWN OFF)
option(SERIAL    OFF)
#option(SERIAL    ON)

# compile flags options
set(OMPFLAGS     "-DOPENMP -fopenmp")
set(CILKFLAGS    "-DCILK -fcilkplus")
set(HGFLAGS      "-DHOMEGROWN -pthread")
set(OPTIMIZATION "-O3 -g")
set(CFLAGS       "-DEDGELONG -DWEIGHTLONG -mcx16 ${OPTIMIZATION} -ldl -std=c++17 -march=native -Wall -Wno-subobject-linkage -DUSEMALLOC -DVERTEXLONG")

if(CLANG)
    set(CMAKE_CXX_COMPILER "clang++")
    set(PFLAGS ${CILKFLAGS})
elseif(CILK)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${CILKFLAGS})
elseif(OPENMP)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${OMPFLAGS})
elseif(HOMEGROWN)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${HGFLAGS})
elseif(SERIAL)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS "")
else()
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${HGFLAGS})
endif()

# set compile flags
set(CMAKE_CXX_FLAGS "${CFLAGS} ${PFLAGS}")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

file(COPY data DESTINATION .)
file(COPY bin/SNAPtoAdj DESTINATION .)
file(COPY bin/to_word2vec.pl DESTINATION .)
file(COPY bin/inverted_index_pam DESTINATION .)

# experiments executables
# add_executable(memory-footprint src/memory-footprint.cpp)
# add_executable(throughput-latency src/throughput-latency.cpp)
add_executable(ut src/ut.cpp)

# -------------------------- Wharf ----------------------------