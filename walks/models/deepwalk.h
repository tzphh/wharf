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
            explicit DeepWalk(dygrl::FlatGraph2* snapshot, types::SampleMethod method)
            {
                this->snapshot = snapshot;
                this->alias_table.resize(0);
                seed = new RandNum(9898676785859);
                this->alias_prefix.resize(0);
                this->alias_table2.resize(0);
                this->chunckSize = config::chunk_size;
                // this->sample_method = method;
            }

            ~DeepWalk()
            {
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

            /**
            * @brief Propose next vertex given current state.
            *
            * @param vertex - current walker state
            *
            * @return - proposed vertex
            */
            types::Vertex propose_vertex(const types::State& state, const types::SampleMethod& method = types::SampleMethod::Naive) final
            {
                switch (method)
                {
                    case types::SampleMethod::Alias:
                        return this->biased_propose_vertex(state);
                    case types::SampleMethod::Reject:
                        return this->reject_propose_vertex(state);
                    case types::SampleMethod::Reservoir:
                        return this->reservoir_propose_vertex(state);
                    case types::SampleMethod::Chunk:
                        return this->chunk_propose_vertex(state);
                    default:
                        return this->naive_propose_vertex(state);
                }
                return 0;
            }

            types::Vertex naive_propose_vertex(const types::State& state) 
            {
                auto neighbors = this->snapshot->neighbors2(state.first);
                const auto& neighbor_list = std::get<0>(neighbors);
                auto degree = std::get<2>(neighbors);
                auto vertex = state.first;

                if (degree == 0) {
                    return vertex;
                }
                auto pos =  seed->iRand(static_cast<uint32_t>(degree));
                return neighbor_list[pos];
            }


            types::Vertex biased_propose_vertex(const types::State& state) 
            {
                auto neighbors = this->snapshot->neighbors2(state.first);
                auto degree = std::get<2>(neighbors);
                auto vertex = state.first;

                if (degree == 0) {
                    return vertex;
                }

                auto& neighbor_list = std::get<0>(neighbors);
                auto pos =  seed->iRand(static_cast<uint32_t>(degree));
                auto prob = seed->dRand();
                
                auto& alias_entry = alias_table[state.first][pos];
                if (prob <= alias_entry.probability) {
                    vertex = neighbor_list[pos];     
                } else {
                    vertex = alias_entry.second;
                }

                return vertex;
            }

            types::Vertex reject_propose_vertex(const types::State& state) 
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
                    if (rej < weight::get_weight(std::get<1>(neighbors)[pos])) {
                        return std::get<0>(neighbors)[pos];
                    } 
                    else {
                        pos =  seed->iRand(static_cast<uint32_t>(degree));
                        rej =  seed->iRand(static_cast<uint32_t>(config::weight_boundry));
                    }
                }
                return 0;
            }

            virtual types::Vertex reservoir_propose_vertex(const types::State& state) 
            {
                const auto neighbors = this->snapshot->neighbors2(state.first);
                const auto& neighbor_list = std::get<0>(neighbors);
                const auto& weight_list = std::get<0>(neighbors);
                const auto degree = std::get<2>(neighbors);
                auto vertex = state.first;

                if (degree == 0) {
                    return vertex;
                }
                types::Weight sum = weight::get_weight(weight_list[0]);
                size_t candidate = 0;
                for (size_t i = 1; i < degree; i++) {
                    sum += weight::get_weight(weight_list[i]);
                    auto r = seed->dRand();
                    r = seed->dRand();
                    auto prob = static_cast<double>(weight::get_weight(weight_list[i])) / sum;
                    if (r < prob) {
                        candidate = i;
                        break;
                    }
                }

                return neighbor_list[candidate];
            }


            types::Vertex chunk_propose_vertex(const types::State& state) 
            {
                auto neighbors = this->snapshot->neighbors2(state.first);   
                auto degree = std::get<2>(neighbors);
                auto vertex = state.first;
                if (degree == 0) {
                    return vertex;
                }
                
                auto total_weight = this->alias_prefix[vertex].back();
                if (total_weight == 0) {
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

                if (off >= this->alias_table2[vertex].size()) {
                    return vertex;
                }
                if (prob <= alias_table2[state.first][off][pos].probability) {
                    vertex = std::get<0>(neighbors)[pos + off * chunckSize];     
                } else {
                    vertex = alias_table2[state.first][off][pos].second;
                }
                return vertex;
            }
            /**
             * @brief Build total alias table
             */
            void build_sample_structure() 
            {
                auto nverts = this->snapshot->size();
                this->alias_table.resize(nverts);
                this->alias_table2.resize(nverts);
                this->alias_prefix.resize(nverts);
                parallel_for(0, nverts, [&](size_t i) {
                    this->build_sample_structure_single(i);
                    this->build_sample_structure_single2(i);
                });
            
                return;
            }

            /**
             * @brief Build alias table for a vertex
             * @param vertex Vertex to build alias table for
             */


            // TODO:debug 生成别名表有问题
            void build_sample_structure_single(size_t vert_id) 
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



            void build_sample_structure_single2(size_t vert_id) 
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
                    this->alias_prefix[vert_id].push_back(alias_prefix[vert_id].back() + std::accumulate(real_weights.begin() + chunckSize * i, real_weights.begin() + chunckSize * (i + 1), 0));
                }
                this->alias_prefix[vert_id].push_back(alias_prefix[vert_id].back() + std::accumulate(real_weights.begin() + chunckSize * (chunckCnt - 1), real_weights.end(), 0));

                this->alias_table2[vert_id].resize(chunckCnt);
                for (size_t j = 0; j < chunckCnt; j++) {
                    chunkCur = std::min(chunckSize, degree - (j * chunckSize));
                    totalWeight = alias_prefix[vert_id][j + 1] - alias_prefix[vert_id][j];
                    probabilities.resize(chunkCur);
                    alias_table2[vert_id][j].resize(chunkCur);

                    for (size_t i = 0; i < chunkCur; ++i) {  
                        probabilities[i] = weight::get_weight(weights[j * chunckSize + i]) / totalWeight;
                    }
                    for (size_t i = 0; i < chunkCur; ++i) {
                        this->alias_table2[vert_id][j][i].probability = probabilities[i] * chunkCur;
                        this->alias_table2[vert_id][j][i].second = 0;
                        if (this->alias_table2[vert_id][j][i].probability < 1.0)
                            smaller.push_back(i);
                        else
                            larger.push_back(i);
                    }
                    while (!smaller.empty() && !larger.empty()) {
                        uintV small = smaller.back();
                        smaller.pop_back();
                        uintV large = larger.back();
                        larger.pop_back();
                        this->alias_table2[vert_id][j][small].second = edges[large + j * chunckSize];
                        this->alias_table2[vert_id][j][large].probability -= (1.0 - this->alias_table2[vert_id][j][small].probability);
                        if (this->alias_table2[vert_id][j][large].probability < 1.0)
                            smaller.push_back(large);
                        else
                            larger.push_back(large);
                    }

                    while (!larger.empty()) {
                        int top = larger.back();
                        larger.pop_back();
                            this->alias_table2[vert_id][j][top].probability = 1.0;
                    }
                    while (!smaller.empty()) {
                        int top = smaller.back();
                        smaller.pop_back();
                        this->alias_table2[vert_id][j][top].probability = 1.0;
                    }
                    probabilities.clear();
                    larger.clear();
                    smaller.clear();
                }
            }


        private:
            FlatGraph2* snapshot;
            std::vector<std::vector<types::AliasTable>> alias_table;
            RandNum *seed;
            std::vector<std::vector<std::vector<types::AliasTable>>> alias_table2;
            std::vector<std::vector<types::Weight>> alias_prefix;
            types::Vertex chunckSize;
            // config::SamplerMethod *sampler_type;
    };
}

#endif
