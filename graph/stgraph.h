#ifndef STGRAPH_H
#define STGRAPH_H

#include <graph/api.h>
#include <cuckoohash_map.hh>
#include <pbbslib/utilities.h>

#include <config.h>
#include <pairings.h>
#include <vertex.h>
#include <snapshot.h>

#include <models/deepwalk.h>
#include <models/node2vec.h>
#include <set>

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{

    class STGraph
    {
    public:
        using Graph = aug_map<dygrl::Vertex2>;
        /**
         * @brief STGraph constructor.
         */
        STGraph(long graph_vertices, long graph_edges, uintE* offsets, uintV* edges, uintW* weights, bool free_memory = true) {
            // 1. Initialize memory pool.
            init_memory_pools(graph_vertices, graph_edges);

            // 2、Create an empty vertex2 sequence
            using VertexStruct = std::pair<types::Vertex, VertexEntry2>;
            auto vertices = pbbs::sequence<VertexStruct>(graph_vertices);

            // 3. In parallel construct graph vertices
            parallel_for(0, graph_vertices, [&](long index)
            {
                size_t off = offsets[index];
                size_t deg = ((index == (graph_vertices - 1)) ? graph_edges : offsets[index + 1]) - off;
                auto SE = pbbs::delayed_seq<uintV>(deg, [&](size_t j) { return edges[off + j]; });
                auto SW = pbbs::delayed_seq<uintW>(deg, [&](size_t j) { return weights[off + j]; });

                vector<dygrl::CompressedWalks> vec_compwalk;
                if (deg > 0)
                    vertices[index] = std::make_pair(index, VertexEntry2(types::CompressedEdges(SE, index),
                                                                        types::CompressedEdges(SW, index), 
                                                                        vec_compwalk,
                                                                        new dygrl::SamplerManager(0)));
                else
                    vertices[index] = std::make_pair(index, VertexEntry2(types::CompressedEdges(),
                                                                        types::CompressedEdges(),
                                                                        vec_compwalk,
                                                                        new dygrl::SamplerManager(0)));
            });

            // 4. Construct the graph
            auto replace = [](const VertexEntry2& x, const VertexEntry2& y) { return y; };
            this->graph_tree = Graph::Tree::multi_insert_sorted(nullptr, vertices.begin(), vertices.size(), replace, true);  

            // 5. GC
            if (free_memory)
            {
                pbbs::free_array(offsets);
                pbbs::free_array(weights);
                pbbs::free_array(edges);
            }
		    vertices.clear();
        }

        ~STGraph();
        void generate_initial_random_walks();
        void destroy();
        void destroy_index();
        [[nodiscard]] FlatGraph flatten_graph() const;
        [[nodiscard]] FlatVertexTree flatten_vertex_tree() const;
    
    private:
        Graph graph_tree;   // same name as in Wharf

        /**
         * @brief Initialize memory pools.
         */
        static void init_memory_pools(size_t graph_vertices, size_t graph_edges)
        {
            types::CompressedTreesLists::init();
            compressed_lists::init(graph_vertices);
            Graph::init();
        }
    };

}

#endif