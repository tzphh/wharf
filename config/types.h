#ifndef TYPES_H
#define TYPES_H

// 
namespace types
{
    // Vertex = graph vertex data type
    using Vertex               = uintV;

    // Degree = graph vertex degree data type
    using Degree               = uintE;

    // Weight = graph edge weight data type
    using Weight               = uintW;

    // Neighbors = graph vertex neighbors
    using Neighbors            = std::tuple<Vertex*, Degree, bool>;

    // Neighbors2 = graph vertex neighbors containing weights
    using Neighbors2          = std::tuple<Vertex*, Weight*, Degree, bool>;
    
    // WalkID = unique walk id
    using WalkID               = uint32_t;

    // Position = the position of a vertex in the walk
    using Position             = uint32_t;

    // PairedTriplet = a triplet <WalkID, Position, NextVertex> after encoded with the pairing function
    using PairedTriplet        = uint64_t;

    // State = the state is defined as a pair of two numbers,
    // where the first represents the current vertex and the second contains the extra information
    // (e.g DeepWalk = current vertex, Node2Vec = previously visited vertex by the walker)
    // 支持二阶游走，引入历史状态信息
    using State                = std::pair<Vertex, Vertex>;

    // CompressedEdges = structure (augmented parallel binary tree) that stores compressed edges
    // C树结构
    using CompressedEdges      = edge_plus::treeplus;
    using CompressedTreesLists = edge_plus::edge_list;

    // RandomWalkModelType = different walking models
    enum RandomWalkModelType   { DEEPWALK, NODE2VEC };

    // StartMode = edge sampler initialization strategy
    enum SamplerInitStartegy   { RANDOM, BURNIN, WEIGHT };

    // Global Map of Changes (MoC) = contains starting positions to crop the walk
    // MAV受到影响的映射
    using MapAffectedVertices  = libcuckoo::cuckoohash_map<WalkID, std::tuple<Position, Vertex, bool>>;

    // ChangeAccumulator = accumulator of changes for walk updates
    // 游走更新的累加器
    using ChangeAccumulator    = libcuckoo::cuckoohash_map<Vertex, std::vector<PairedTriplet>>;

    //Alias Table
    using AliasTable = struct {
        prob probability;
        Vertex second;
    };
    
    // sample method
    enum class SampleMethod {
        Naive,
        Reject,
        Reservoir,
        Alias,
        Chunk,
        Its
    };
}

#endif
