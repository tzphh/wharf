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
            /**
            * @brief Determines an initial state of the walker.
            *
            * @param vertex - graph vertex
            *
            * @return - an initial state of the walker
            */
            virtual types::State initial_state(types::Vertex vertex) = 0;

            /**
            * @brief The transition of states is crucial for the walking.
            * Based on the next vertex and the current state we update the state.
            *
            * @param state  - current state of the walker
            * @param vertex - next vertex to go
            *
            * @return - a new state of the walker
            */
            virtual types::State new_state(const types::State& state, types::Vertex vertex) = 0;

            /**
            * @brief Calculates the edge weight based on the current state and the potentially proposed vertex.
            *
            * @param state  - current state of the walker
            * @param vertex - potentially proposed vertex
            *
            * @return - dynamically calculated weight
            */
            virtual float weight(const types::State& state, types::Vertex vertex) = 0;

            /**
             * @brief Propose next vertex given current state.
             *
             * @param vertex - current walker state
             *
             * @return - proposed vertex
             */
            virtual types::Vertex propose_vertex(const types::State& state) = 0;

             /**
             * @brief Biased propose next vertex given current state .
             *
             * @param vertex - current walker state
             *
             * @return - proposed vertex
             */
            virtual types::Vertex biased_propose_vertex(const types::State& state) = 0;

            virtual types::Vertex reject_propose_vertex(const types::State& state) = 0;

            /**
             * @brief Generate alias table. 
             *
             */
            virtual void build_sample_structure() = 0;
            virtual void build_sample_structure_single(size_t i) = 0;
            // virtual std::vector<std::vector<types::AliasTable>>& get_alias_table() = 0;
            //virtual void update_snapshot(dygrl::Snapshot* snapshot) = 0;
    };
}

#endif
