#ifndef RANDOM_WALK_MODEL_H
#define RANDOM_WALK_MODEL_H

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
    * RandomWalkModel is the interface that all random walk models (e.g DeepWalk, Node2Vec, etc.) need to implement.
    */
    class RandomWalkModel
    {
        public:
            virtual types::State initial_state(types::Vertex vertex) = 0;
            virtual types::State new_state(const types::State& state, types::Vertex vertex) = 0;
            virtual float weight(const types::State& state, types::Vertex vertex) = 0;
            virtual types::Vertex propose_vertex(const types::State& state, const types::SampleMethod& method = types::SampleMethod::Naive) = 0;

            // virtual types::Vertex biased_propose_vertex(const types::State& state) = 0;
            // virtual types::Vertex reject_propose_vertex(const types::State& state) = 0;
            // virtual types::Vertex reservoir_propose_vertex(const types::State& state) = 0;
            // virtual void build_sample_structure() = 0;
            // virtual void build_sample_structure_single(size_t i) = 0;

            
            // types::SampleMethod sample_method;
    };
}

#endif
