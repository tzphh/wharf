#ifndef DEEPWALK_HYBRID_SAMPLE_H
#define DEEPWALK_HYBRID_SAMPLE_H

#include <random_walk_model.h>
#include <snapshot.h>
#include <snapshot2.h>
#include <weight.h>
#include <utility.h>
#include <random>  
#include "fast_random.h"

namespace dynamic_graph_representation_learning_with_metropolis_hastings 
{
    class DeepWalkHybridSample : public RandomWalkModel
    {
        public:   
            explicit DeepWalkHybridSample(dygrl::Snapshot* snapshot, types::Vertex chunckSize)
            {
                this->snapshot = snapshot;
                this->alias_table.resize(0);
                this->alias_prefix.resize(0);
                this->chunckSize = chunckSize;
                seed = new RandNum(9898676785859);
            }

            ~DeepWalkHybridSample()
            {
                delete seed;
                this->alias_prefix.clear();
                this->alias_table.clear();
                if (!this->snapshot) {
                    delete this->snapshot;
                }
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

            }

            types::Vertex biased_propose_vertex(const types::State& state) final
            {
                auto neighbors = this->snapshot->neighbors2(state.first);   
                auto degree = std::get<2>(neighbors);
                auto vertex = state.first;
                auto total_weight = this->alias_prefix[vertex].back();
                if (total_weight == 0 || degree == 0) {
                    return vertex;
                }
                
                uintV chunckCur = degree;
                auto off = 0;
                uintV abs = 0;
                if (degree > chunckSize) {
                    abs= seed->iRand(static_cast<uint32_t>(total_weight));
                    auto poss = std::upper_bound(this->alias_prefix[vertex].begin(), this->alias_prefix[vertex].end(), abs);
                    off = static_cast<uintV>(std::distance(this->alias_prefix[vertex].begin(), poss - 1));
                    chunckCur = std::min(chunckSize, degree - (off * chunckSize));
                }

                uintV pos = seed->iRand(static_cast<uint32_t>(chunckCur));
                auto prob = seed->dRand();

                if (off >= this->alias_table[vertex].size()) {
                    return vertex;
                }
                if (prob <= alias_table[state.first][off][pos].probability) {
                    vertex = std::get<0>(neighbors)[pos + off * chunckSize];     
                } else {
                    vertex = alias_table[state.first][off][pos].second;
                }
                return vertex;
            }

            void build_sample_structure() final
            {
                auto nverts = this->snapshot->size();
                this->alias_table.resize(nverts);
                this->alias_prefix.resize(nverts);
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
                uintV chunckCnt = 0;
                uintV chunkCur = 0;
                std::vector<prob> probabilities;
                std::vector<prob> prefix_weights;
                std::vector<prob> real_weights;
                std::vector<uintV> smaller, larger;

                auto neighbors = this->snapshot->neighbors2(vert_id);
                auto edges = std::get<0>(neighbors);
                auto weights = std::get<1>(neighbors);
                auto degree = std::get<2>(neighbors);
                prob totalWeight = 0.0;

                if (degree == 0) return;
                probabilities.resize(degree);
                prefix_weights.resize(degree + 1);  // weights[i] = prefix_weights[i + 1] - prefix_weights[i];
                real_weights.resize(degree);
                prefix_weights[0] = 0;

                this->alias_prefix[vert_id].resize(0);

                for (size_t i = 0; i < degree; i++) {
                    prefix_weights[i + 1] = prefix_weights[i] + weight::get_weight(weights[i]);
                    real_weights[i] = weight::get_weight(weights[i]);
                }
                chunckCnt = (degree / chunckSize) + 1;
                this->alias_prefix[vert_id].push_back(0);
                for (size_t i = 0; i < chunckCnt - 1; i++) {
                    // this->alias_prefix[vert_id].push_back(prefix_weights[chunckSize * (i + 1)]);
                    this->alias_prefix[vert_id].push_back(alias_prefix[vert_id].back() + std::accumulate(real_weights.begin() + chunckSize * i, real_weights.begin() + chunckSize * (i + 1), 0));
                }
                // this->alias_prefix[vert_id].push_back(prefix_weights[degree]);
                this->alias_prefix[vert_id].push_back(alias_prefix[vert_id].back() + std::accumulate(real_weights.begin() + chunckSize * (chunckCnt - 1), real_weights.end(), 0));
                // TODO: check prefix correctness

                this->alias_table[vert_id].resize(chunckCnt);
                for (size_t j = 0; j < chunckCnt; j++) {
                    chunkCur = std::min(chunckSize, degree - (j * chunckSize));
                    // totalWeight = prefix_weights[j + 1] - prefix_weights[j];
                    totalWeight = alias_prefix[vert_id][j + 1] - alias_prefix[vert_id][j];
                    probabilities.resize(chunkCur);
                    alias_table[vert_id][j].resize(chunkCur);

                    for (size_t i = 0; i < chunkCur; ++i) {  
                        probabilities[i] = weight::get_weight(weights[j * chunckSize + i]) / totalWeight;
                    }
                    for (size_t i = 0; i < chunkCur; ++i) {
                        this->alias_table[vert_id][j][i].probability = probabilities[i] * chunkCur;
                        this->alias_table[vert_id][j][i].second = 0;
                        if (this->alias_table[vert_id][j][i].probability < 1.0)
                            smaller.push_back(i);
                        else
                            larger.push_back(i);
                    }
                    while (!smaller.empty() && !larger.empty()) {
                        uintV small = smaller.back();
                        smaller.pop_back();
                        uintV large = larger.back();
                        larger.pop_back();
                        this->alias_table[vert_id][j][small].second = edges[large + j * chunckSize];
                        this->alias_table[vert_id][j][large].probability -= (1.0 - this->alias_table[vert_id][j][small].probability);
                        if (this->alias_table[vert_id][j][large].probability < 1.0)
                            smaller.push_back(large);
                        else
                            larger.push_back(large);
                    }

                    while (!larger.empty()) {
                        int top = larger.back();
                        larger.pop_back();
                            this->alias_table[vert_id][j][top].probability = 1.0;
                    }
                    while (!smaller.empty()) {
                        int top = smaller.back();
                        smaller.pop_back();
                        this->alias_table[vert_id][j][top].probability = 1.0;
                    }
                    probabilities.clear();
                    larger.clear();
                    smaller.clear();
                }
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
            std::vector<std::vector<std::vector<types::AliasTable>>> alias_table;
            std::vector<std::vector<types::Weight>> alias_prefix;
            types::Vertex chunckSize;
            RandNum *seed;
    };
}

#endif
