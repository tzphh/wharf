#ifndef DEEPWALK_ALIAS_SAMPLE_H
#define DEEPWALK_ALIAS_SAMPLE_H

#include <random_walk_model.h>
#include <snapshot.h>
#include <snapshot2.h>
#include <weight.h>
#include <utility.h>
#include <random>  
#include <cassert>
#include "fast_random.h"

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    class DeepWalkAliasSample : public RandomWalkModel
    {
        public:
            explicit DeepWalkAliasSample(dygrl::Snapshot* snapshot)
            {
                this->snapshot = snapshot;
                this->alias_table.resize(0);
                seed = new RandNum(9898676785859);
            }

            ~DeepWalkAliasSample()
            {
                delete seed;
                this->alias_table.clear();
                if (!this->snapshot)
                    delete this->snapshot;
            }

            types::State initial_state(types::Vertex vertex) final
            {
                return types::State(vertex, vertex);
            }

            types::State new_state(const types::State& state, types::Vertex vertex) final
            {
                return types::State(vertex, vertex);
            }

            float weight(const types::State& state, types::Vertex vertex) final
            {
                return 1.0;
            }

            types::Vertex propose_vertex(const types::State& state) final
            {
                return 0;
            }

            // 别名采样
            types::Vertex biased_propose_vertex(const types::State& state) final
            {
                
                assert(this->alias_table.size() == this->snapshot->size());
                auto neighbors = this->snapshot->neighbors2(state.first);   
                auto degree = std::get<2>(neighbors);
                auto vertex = state.first;
                
                // std::cout <<"current vert is" << state.first << std::endl;

                if (degree == 0) {
                    return vertex;
                }
                auto pos =  seed->iRand(static_cast<uint32_t>(degree));
                auto prob = seed->dRand();
                
                if (prob <= alias_table[state.first][pos].probability) {
                    vertex = std::get<0>(neighbors)[pos];     
                } else {
                    vertex = alias_table[state.first][pos].second;
                }

                //std::cout << "current sample vertex is" << vertex << std::endl;
                // std::cout <<"current vert is" << state.first << " ,sample " << vertex << ::endl;
                return vertex;
            }

            /**
             * @brief Build total alias table
             */
            void build_sample_structure() final
            {
                auto nverts = this->snapshot->size();
                this->alias_table.resize(nverts);
                for (size_t i = 0; i < nverts; i++) {
                    this->build_sample_structure_single(i);
                }
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
                    // std::cout << "real_weights is " << real_weights[i] << std::endl;
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
                    // delete this->snapshot;  // 假设 snapshot 是通过 new 创建的
                    this->snapshot = snapshot;  // 更新成员变量
                }
                return;
            }

        private:
            Snapshot* snapshot;
            std::vector<std::vector<types::AliasTable>> alias_table;
            RandNum *seed;
    };

}


#endif
