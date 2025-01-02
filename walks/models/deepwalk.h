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
                    case types::SampleMethod::Its:
                        return this->its_propose_vertex(state);
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

            types::Vertex chunk_propose_vertex(const types::State& state) {
                // 提前引用需要频繁访问的数据，减少缓存未命中
                const auto& alias_prefix_vertex = this->alias_prefix[state.first];
                const auto& alias_table = this->alias_table2[state.first];
                const auto& neighbors_data = this->snapshot->neighbors2(state.first);
                const auto& neighbors = std::get<0>(neighbors_data);
                uintV degree = std::get<2>(neighbors_data);

                // 快速返回条件，减少不必要的后续操作
                if (degree == 0 || alias_prefix_vertex.empty() || alias_prefix_vertex.back() == 0) {
                    return state.first;
                }

                uintV offset = 0, chunk_cur = degree;

                // 静态预测分支优化与预取指令
                if (__builtin_expect(degree > chunckSize, 1)) {
                    uint32_t rand_abs = seed->iRand(static_cast<uint32_t>(alias_prefix_vertex.back()));

                    // 使用手写二分搜索替代 std::upper_bound，提高性能
                    uintV low = 0, high = alias_prefix_vertex.size() - 1;
                    while (low < high) {
                        uintV mid = (low + high) / 2;
                        if (alias_prefix_vertex[mid] > rand_abs) {
                            high = mid;
                        } else {
                            low = mid + 1;
                        }
                    }
                    offset = low - 1;

                    // 使用 std::min 避免复杂条件判断
                    chunk_cur = std::min(chunckSize, degree - offset * chunckSize);
                }

                // 直接计算随机位置和概率，减少函数调用
                uintV pos = seed->iRand(static_cast<uint32_t>(chunk_cur));
                double prob = seed->dRand();

                // 检查 offset 是否在有效范围内，避免访问越界
                if (__builtin_expect(offset < alias_table.size(), 1)) {
                    const auto& alias_entry = alias_table[offset][pos];

                    // 使用 SIMD 优化条件判断
                    return (prob <= alias_entry.probability) ? neighbors[pos + offset * chunckSize] : alias_entry.second;
                }

                return state.first;
            }


            types::Vertex its_propose_vertex(const types::State& state) {
                // 提前引用需要频繁访问的数据，减少缓存未命中
                const auto& neighbors_data = this->snapshot->neighbors(state.first);
                const auto& weight_prefix_table = this->weight_prefix[state.first];
                const auto& neighbors = std::get<0>(neighbors_data);
                const auto degree = std::get<1>(neighbors_data);
                auto vertex = state.first;

                if (degree == 0) {
                    return vertex;
                }

                auto total_weight = weight_prefix_table.back();
                auto rand_abs = seed->iRand(static_cast<uint32_t>(total_weight));
                auto abs = std::lower_bound(weight_prefix_table.begin(), weight_prefix_table.end(), rand_abs);
                auto pos = static_cast<size_t>(std::distance(weight_prefix_table.begin(), abs - 1));
                
                return neighbors[pos];
            }

            /**
             * @brief Build total alias table
             */
            void build_sample_structure() 
            {
                auto nverts = this->snapshot->size();

                switch (config::sample_method) 
                {
                    case types::SampleMethod::Alias:
                    {
                        this->alias_table.resize(nverts);
                        parallel_for(0, nverts, [&](size_t i) {
                            this->build_sample_structure_single(i);
                        });
                        break;
                    }
                    case types::SampleMethod::Chunk:
                    {
                        this->alias_table2.resize(nverts);
                        this->alias_prefix.resize(nverts);
                        parallel_for(0, nverts, [&](size_t i) {
                            this->build_sample_structure_single2(i);
                        });
                        break;
                    }
                    case types::SampleMethod::Its:
                    {
                        this->weight_prefix.resize(nverts);
                        parallel_for(0, nverts, [&](size_t i) {
                            this->build_sample_structure_single3(i);
                        });
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }
                return;
            }

            /**
             * @brief Build alias table for a vertex
             * @param vertex Vertex to build alias table for
             */
            // for alias sample
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


            // for chunk sample
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


            // for its sample
            void build_sample_structure_single3(size_t vert_id) 
            {
                auto neighbors = this->snapshot->neighbors2(vert_id);
                // auto edges = std::get<0>(neighbors);
                auto weights = std::get<1>(neighbors);
                auto degree = std::get<2>(neighbors);

                std::vector<types::Weight> real_weights(degree, 0);
                parallel_for(0, degree, [&](size_t i) {
                    real_weights[i] = weight::get_weight(weights[i]);
                });

                weight_prefix[vert_id].resize(degree + 1);
                weight_prefix[vert_id][0] = 0;
                for (size_t i = 0; i < degree; i++) {
                    weight_prefix[vert_id][i + 1] = weight_prefix[vert_id][i] + real_weights[i];
                }

                return;
            }
        private:
            FlatGraph2* snapshot;
            std::vector<std::vector<types::AliasTable>> alias_table;
            RandNum *seed;
            std::vector<std::vector<std::vector<types::AliasTable>>> alias_table2;    // for chunk sample
            std::vector<std::vector<types::Weight>> alias_prefix;                     // for chunk sample
            types::Vertex chunckSize;
            // config::SamplerMethod *sampler_type;
            std::vector<std::vector<types::Weight>> weight_prefix;                    // for its sample
 
    };
}

#endif
