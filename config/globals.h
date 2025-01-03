#ifndef GLOBALS_H
#define GLOBALS_H

#include "types.h"
// #include "utility.h"

namespace config
{
    // determines the number of walks per vertex to be generated by the walkers
    uint8_t walks_per_vertex   = 1;

    // determines the length of one random walk
    uint8_t walk_length        = 80;

    // determines the type of random walk model to use
    auto random_walk_model     = types::RandomWalkModelType::DEEPWALK;

    // determines parameter P for node2vec model
    float paramP               = 4;

    // determines parameter Q for node2vec model
    float paramQ               = 1;

    // determines the initialization strategy for metropolis hastings samplers
    auto sampler_init_strategy = types::SamplerInitStartegy::WEIGHT;

    // random number generator
    // auto random                = utility::Random(std::time(nullptr));

    // use range search when searching in walk-trees
    auto range_search_mode     = true;

    // determines whether to produce and update the walks in a deterministic way
    auto deterministic_mode    = true;

	// serial or parallel
	auto parallel_merge_wu     = true;

	// merge every x batches
	int merge_frequency = 1;

    // determines whether to use the biased method for sampling, which is desiged for weighted graphs
    bool biased_sampling       = true;

    // determines the maximum weight of an edge
    size_t weight_boundry     = 100000;

    // total vertexs in this graph
    size_t graph_vertices    = 20;

    // max batch num in this graph
    size_t max_batch_num    = 50;

    // sample strategy
    types::SampleMethod sample_method = types::SampleMethod::Naive;

    // chunk size
    size_t chunk_size = 8;
}

#endif
