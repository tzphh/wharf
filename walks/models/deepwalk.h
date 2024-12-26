#ifndef DEEPWALK_H
#define DEEPWALK_H

#include <random_walk_model.h>
#include <snapshot.h>
#include <snapshot2.h>
#include <weight.h>
#include <utility.h>
#include <random>  
#include "fast_random.h"

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
     * @brief DeepWalk random walk model implementation.
     * @details https://dl.acm.org/doi/abs/10.1145/2623330.2623732
     */
    class DeepWalk : public RandomWalkModel
    {
        public:
            /**
             * @brief DeepWalk constructor.
             *
             * @param snapshot - graph snapshot
             */
            explicit DeepWalk(dygrl::FlatGraph2* snapshot, types::SampleMethod method)
            {
                this->snapshot = snapshot;
                this->alias_table.resize(0);
                seed = new RandNum(9898676785859);
                this->sample_method = method;
            }

            /**
            * @brief DeepWalk destructor.
            */
            ~DeepWalk()
            {
                this->alias_table.clear();
                if (!this->snapshot)
                    delete this->snapshot;
            }

            /**
            * @brief Determines an initial state of the walker.
            *
            * @param vertex - graph vertex
            *
            * @return - an initial state of the walker
            */
            types::State initial_state(types::Vertex vertex) final
            {
                return types::State(vertex, vertex);
            }

            /**
            * @brief The transition of states is crucial for the walking.
            * Based on the next vertex and the current state we update the state.
            *
            * @param state  - current state of the walker
            * @param vertex - next vertex to go
            *
            * @return - a new state of the walker
            */
            types::State new_state(const types::State& state, types::Vertex vertex) final
            {
                return types::State(vertex, vertex);
            }

            /**
            * @brief Calculates the edge weight based on the current state and the potentially proposed vertex.
            *
            * @param state  - current state of the walker
            * @param vertex - potentially proposed vertex
            *
            * @return - dynamically calculated weight
            */
            float weight(const types::State& state, types::Vertex vertex) final
            {
                return 1.0;
            }

            /**
            * @brief Propose next vertex given current state.
            *
            * @param vertex - current walker state
            *
            * @return - proposed vertex
            */
            types::Vertex propose_vertex(const types::State& state) final
            {
                auto random                = utility::Random(std::time(nullptr));
                auto neighbors = this->snapshot->neighbors(state.first);   // 三元组 (邻居数组, 邻居数, 是否需要释放内存)
                // auto vertex    = std::get<0>(neighbors)[config::random.irand(std::get<1>(neighbors))];  // todo: check 度数是否为0
                // degree为0时下一跳为源顶点
                auto vertex    = std::get<1>(neighbors) == 0 ? state.first : std::get<0>(neighbors)[random.irand(std::get<1>(neighbors))];  
                

                if (std::get<2>(neighbors)) pbbs::free_array(std::get<0>(neighbors));

                return vertex;
            }

            /**
            * @brief Biased propose next vertex given current state.
            *
            * @param state - current walker state
            *
            * @return - proposed vertex
            */
            // 别名cia
            types::Vertex biased_propose_vertex(const types::State& state) final
            {
                // assert(this->alias_table.size() == this->snapshot->size());
                //auto neighbors = this->snapshot->neighbors2(state.first);   

                // auto
                // auto neightors = this->snapshot[state.first].neighbors2;

                auto neighbors = this->snapshot->neighbors2(state.first);

                auto degree = std::get<2>(neighbors);
                auto vertex = state.first;
                
                // std::cout <<"current vert is" << state.first << std::endl;

                if (degree == 0) {
                    // cout << "degree is 0" << endl;
                    // std::cout <<"current vert is" << state.first << " ,sample " << vertex << ::endl;
                    return vertex;
                }
                // auto random                = utility::Random(std::time(nullptr));
                auto pos =  seed->iRand(static_cast<uint32_t>(degree));
                auto prob = seed->dRand();


                // std::cout << "current vertex is" << vertex << std::endl;
                // std::cout << state.first <<" pos: " << pos << " prob  "<<prob << std::endl;
                // for (size_t i = 0; i < degree; i++) {
                //     std::cout << "neighbor is " << std::get<0>(neighbors)[i] << " weight is" <<std::get<1>(neighbors)[i]<<std::endl;
                // }

                
                if (prob <= alias_table[state.first][pos].probability) {
                    vertex = std::get<0>(neighbors)[pos];     
                } else {
                    vertex = alias_table[state.first][pos].second;
                }

                //std::cout << "current sample vertex is" << vertex << std::endl;
                // std::cout <<"current vert is" << state.first << " ,sample " << vertex << ::endl;
                return vertex;
            }



            types::Vertex reject_propose_vertex(const types::State& state) final 
            {
                // seed = new RandNum(742429651);
                auto neighbors = this->snapshot->neighbors2(state.first);
                auto degree = std::get<2>(neighbors);
                auto vertex = state.first;

                if (degree == 0) {
                    return vertex;
                }
        
                auto pos =  seed->iRand(static_cast<uint32_t>(degree));
                auto rej =  seed->iRand(static_cast<uint32_t>(config::weight_boundry));
                while (true) {
                    if (rej < std::get<1>(neighbors)[pos]) {
                        return std::get<0>(neighbors)[pos];
                    } 
                    else {
                        pos =  seed->iRand(static_cast<uint32_t>(degree));
                        rej =  seed->iRand(static_cast<uint32_t>(config::weight_boundry));
                    }
                }

                return 0;
            }

            virtual types::Vertex reservoir_propose_vertex(const types::State& state) final
            {
                return 0;
            }
            /**
             * @brief Build total alias table
             */
            void build_sample_structure() final
            {
                auto nverts = this->snapshot->size();
                this->alias_table.resize(nverts);

                parallel_for(0, nverts, [&](size_t i) {
                    this->build_sample_structure_single(i);
                });
            
                return;
            }

            /**
             * @brief Build alias table for a vertex
             * @param vertex Vertex to build alias table for
             */


            // TODO:debug 生成别名表有问题
            void build_sample_structure_single(size_t vert_id) final
            {
                std::vector<prob> probabilities;
                std::vector<prob> real_weights;
                std::vector<uintV> smaller, larger;
                std::vector<prob> weight_list;

                auto neighbors = this->snapshot->neighbors2(vert_id);   
                auto edges = std::get<0>(neighbors);
                auto weights = std::get<1>(neighbors);
                auto degree = std::get<2>(neighbors);
                prob totalWeight = 0.0;

                if (degree == 0) return;

                probabilities.resize(degree);
                real_weights.resize(degree);
                this->alias_table[vert_id].resize(degree);

                parallel_for(0, degree, [&](size_t i) {
                    real_weights[i] = weight::get_weight(weights[i]);
                });
                
                // TODO：parallelize
                for (size_t i = 0; i < degree; i++) {
                    totalWeight += real_weights[i];
                }

                for (size_t i = 0; i < degree; i++) {
                    probabilities[i] = real_weights[i] / totalWeight;
                    //std::cout << weights[i] << " " << real_weights[i] << " " << totalWeight << " " << probabilities[i] << std::endl;
                }
            
                for (size_t i = 0; i < degree; i++) {
                    alias_table[vert_id][i].probability = probabilities[i] * degree;
                    alias_table[vert_id][i].second = 0;
                    if (alias_table[vert_id][i].probability < 1.0) {
                        smaller.push_back(i);
                    } else {
                        larger.push_back(i);
                    }
                }

                size_t n_smaller, n_larger;
                while (!smaller.empty() && !larger.empty()) {
                    n_smaller = smaller.back();
                    smaller.pop_back();
                    n_larger = larger.back();
                    larger.pop_back();

                    alias_table[vert_id][n_smaller].second = edges[n_larger];
                    alias_table[vert_id][n_larger].probability -= (1.0 -  alias_table[vert_id][n_smaller].probability);
                    if (alias_table[vert_id][n_larger].probability < 1.0) {
                        smaller.push_back(n_larger);
                    } 
                    else {
                        larger.push_back(n_larger);
                    }   
                }

                while (!larger.empty()) {
                    int top = larger.back();
                    larger.pop_back();
                    alias_table[vert_id][top].probability = 1.0;
                }
                while (!smaller.empty()) {
                    int top = smaller.back();
                    smaller.pop_back();
                    alias_table[vert_id][top].probability = 1.0;
                }
                probabilities.clear();
                larger.clear();
                smaller.clear();
                return;
            }

             std::vector<std::vector<types::AliasTable>>& get_alias_table() 
             {
                return this->alias_table;
             }

            void update_snapshot(dygrl::FlatGraph2* snapshot) 
            {
                if (this->snapshot != snapshot) {
                    this->snapshot = snapshot;  // 更新成员变量
                }
                return;
            }

        private:
            FlatGraph2* snapshot;
            std::vector<std::vector<types::AliasTable>> alias_table;
            RandNum *seed;
            // config::SamplerMethod *sampler_type;
    };
}

#endif
