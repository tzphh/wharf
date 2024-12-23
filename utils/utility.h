#ifndef UTILITY_H
#define UTILITY_H

#include <rmat_util.h>
#include <globals.h>

namespace utility
{
    /**
     * @brief Converts B to GB.
     *
     * @param bytes - size in B
     *
     * @return size in GB
     */
    double GB(size_t bytes)
    {
        double gb = bytes;
        gb /= 1024.0;
        gb /= 1024.0;
        gb /= 1024.0;

        return gb;
    }

    /**
     * @brief Converts B to MB.
     *
     * @param bytes - size in B
     *
     * @return size in MB
     */
    double MB(size_t bytes)
    {
        double mb = bytes;
        mb /= 1024.0;
        mb /= 1024.0;

        return mb;
    }

    /**
    * @brief Generates batch of edges for insertion or deletion.
    *
    * @param edges_number - number of edges to generate
    * @param vertices_number - number of vertices in the graph
    * @param self_loops - determines whether self loops are allowed
    * @param directed - determines whether it needs to generate directed or undirected edges
    * @param a
    * @param b
    * @param c
    * @param run_seq - determines whether to run part of the code sequantially
    * @param weighted - determines whether to generate weighted edges
    *
    * @return - batch of generated edges
    */
    auto generate_batch_of_edges
    (
        size_t edges_number,
        size_t vertices_number,
		size_t batch_seed = 0,
        bool self_loops = false,
        bool directed = true,
        bool weighted = false,
        double a = 0.5,
        double b = 0.2,
        double c = 0.1,
        bool run_seq = false,
        size_t weight_boundry = config::weight_boundry
    )
    {
        #ifdef WHARF_TIMER
            timer timer("Utility::GenerateBatchOfEdges");
        #endif

        using Edge = std::tuple<uintV, uintV>;  // (vertex1, vertex2)

        // 1. Set up for the edge generation
        auto rand              = pbbs::random(batch_seed);
        size_t graph_size_pow2 = 1 << (pbbs::log2_up(vertices_number) - 1);
        auto rmat              = rMat<unsigned int>(graph_size_pow2, rand.ith_rand(0), a, b, c);
	    auto edges             = (directed) ? pbbs::new_array<Edge>(edges_number) : pbbs::new_array<Edge>(2 * edges_number);

        // 2. Generate edges in parallel
        parallel_for(0, edges_number, [&] (size_t i)
        {
            edges[i] = rmat(i);
        });

        if (!directed)
        {
            parallel_for(0, edges_number, [&] (size_t i)
            {
                edges[i + edges_number] = std::make_tuple(std::get<1>(edges[i]), std::get<0>(edges[i]));;
            });
        }

        // 3. Sort edges by source
        auto edges_sorted = (directed) ? pbbs::make_range(edges, edges + edges_number) : pbbs::make_range(edges, edges + 2 * edges_number);

        auto node_bits    = pbbs::log2_up(graph_size_pow2);

        auto edge_to_long = [graph_size_pow2, node_bits](Edge e) -> size_t {
            return (static_cast<size_t>(std::get<0>(e)) << node_bits) + static_cast<size_t>(std::get<1>(e));
        };

        size_t bits = 2 * node_bits;

        if (graph_size_pow2 <= (edges_sorted.size() * pbbs::log2_up(edges_sorted.size())))
        {
            pbbs::integer_sort_inplace(edges_sorted, edge_to_long, bits);
        }
        else
        {
            pbbs::sample_sort_inplace(edges_sorted, std::less<>());
        }

        // 4. Remove duplicate edges
        Edge* generated_edges = nullptr;

        // Remove duplicate edges
        auto bool_seq = pbbs::delayed_seq<bool>(edges_sorted.size(), [&] (size_t i)
        {
            if (!self_loops && std::get<0>(edges_sorted[i]) == std::get<1>(edges_sorted[i])) return false;
            return (i == 0 || edges_sorted[i] != edges_sorted[i - 1]);
        });

        auto pack = pbbs::pack(edges_sorted, bool_seq, run_seq ? pbbs::fl_sequential : pbbs::no_flag);
        auto edges_generated = pack.size();
        generated_edges = pack.to_array();

        // 遍历输出generated_edges
        // std::cout << "generated_edges:" << std::endl;
        // for (size_t i = 0; i < edges_generated; i++)
        // {
        //     std::cout << std::get<0>(generated_edges[i]) << " " << std::get<1>(generated_edges[i]) << std::endl;
        // }

        // generate weights
        if (weighted) {
            pbbs::random r(batch_seed);
            auto weights = pbbs::new_array<uintW>(edges_generated);     
            parallel_for(0, edges_generated, [&] (size_t i)
            {
                weights[i] = std::get<1>(generated_edges[i]) * config::graph_vertices + r.ith_rand(i) % weight_boundry;
            });
            return std::make_tuple(generated_edges, edges_generated, weights);
        }

        return std::make_tuple(generated_edges, edges_generated, static_cast<uintW*>(nullptr));
    }

    


    auto generate_edges_from_file(const std::string& filename, bool weighted = false) {
        std::ifstream file(filename);
        
        // 检查文件是否成功打开
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return std::make_tuple(std::vector<std::tuple<uintV, uintV>>(), size_t(0), std::vector<uintW>());
        }

        std::vector<std::tuple<uintV, uintV>> edges;
        std::vector<uintW> weights;
        
        uintV src, dst;
        uintW weight;
        size_t edge_count = 0;  // 明确使用 size_t 类型

        // 读取文件中的每一行，解析 src, dst 和 weight
        while (file >> src >> dst) {
            // 读取权重
            if (weighted) {
                file >> weight;
                weights.push_back(weight);
            }
            edges.push_back(std::make_tuple(src, dst));
            edge_count++;
        }

        file.close();

        // 返回生成的边及其数量
        if (weighted) {
            return std::make_tuple(edges, edge_count, weights);
        } else {
            return std::make_tuple(edges, edge_count, std::vector<uintW>());
        }
    }

    
    /**
     * @brief Random number generator.
     * @details http://xoroshiro.di.unimi.it/#shootout
     */
    class Random
    {
        public:
            uint64_t rng_seed0, rng_seed1;

            explicit Random(uint64_t seed)
            {
                for (int i = 0; i < 2; i++)
                {
                    long long z = seed += UINT64_C(0x9E3779B97F4A7C15);

                    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
                    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);

                    if (i == 0)
                        rng_seed0 = z ^ (z >> 31);
                    else
                        rng_seed1 = z ^ (z >> 31);
                }
            }

            void reinit(uint64_t seed)
            {
                for (int i = 0; i < 2; i++)
                {
                    long long z = seed += UINT64_C(0x9E3779B97F4A7C15);

                    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
                    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);

                    if (i == 0)
                        rng_seed0 = z ^ (z >> 31);
                    else
                        rng_seed1 = z ^ (z >> 31);
                }
            }

            static inline uint64_t rotl(const uint64_t x, int k)
            {
                return (x << k) | (x >> (64 - k));
            }

            uint64_t lrand()
            {
                const uint64_t s0 = rng_seed0;
                uint64_t s1 = rng_seed1;

                const uint64_t result = s0 + s1;
                s1 ^= s0;

                rng_seed0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
                rng_seed1 = rotl(s1, 36);

                return result;
            }

            double drand()
            {
                const union un
                {
                    uint64_t i;
                    double d;
                }
                a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};

                return a.d - 1.0;
            }

            int irand(int max) { return lrand() % max; }

            int irand(int min, int max) { return lrand() % (max - min) + min; }
    };
}

#endif
