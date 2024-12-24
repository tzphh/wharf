#ifndef SNAPSHOT2_H
#define SNAPSHOT2_H
#include "snapshot.h"
#include "types.h"
#include "pbbslib/sequence.h"
#include "vertex.h"

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
     * @brief FlatVertexTree stores vertex2 entries in an array.
     * This allows for O(1) access to vertex content and promises improved cache locality.
     */
    class FlatVertexTree2 : public Snapshot
    {
        public:
            /**
             * @brief FlatVertexTree constructor.
             * 
             * @param vertices - number of vertices in a graph
             */
            explicit FlatVertexTree2(types::Vertex vertices)
            {
                this->snapshot = pbbs::sequence<VertexEntry2>(vertices);
            }

            /**
            * @brief FlatVertexTree subscript operator overloading.
            *
            * @param vertex - graph vertex
            *
            * @return - snapshot entry for a given vertex
            */
            VertexEntry2& operator[](types::Vertex vertex)
            {
                return this->snapshot[vertex];
            }

            /**
            * @brief FlatVertexTree size.
            *
            * @return - number of entries in a snapshot
            */
            types::Vertex size() final
            {
                return this->snapshot.size();
            }

            /**
            * @brief FlatVertexTree size in bytes. The memory footprint of a snapshot.
            *
            * @return - size of a snapshot in bytes
            */
            size_t size_in_bytes() final
            {
                return this->snapshot.size() * sizeof(this->snapshot[0]);
            }

            /**
            * @brief Given a graph vertex it returns its neighbors ,weights and degree.
            *
            * @param vertex - graph vertex
            *
            * @return - neighbors and degree of a given vertex
            */
            types::Neighbors neighbors(types::Vertex vertex) final
            {
                auto neighbors = snapshot[vertex].compressed_edges.get_edges(vertex);
                auto degrees   = snapshot[vertex].compressed_edges.degree();

                return std::make_tuple(neighbors, degrees, true);
            }

            types::Neighbors2 neighbors2(types::Vertex vertex) final
            {
                // TODO: 优化时间复杂度
                // TODO: wharf 无法提供线性时间内访问图数据的能力
                auto neighbors = snapshot[vertex].compressed_edges.get_edges(vertex);
                auto weights   = snapshot[vertex].compressed_weights.get_edges(vertex);
                auto degrees   = snapshot[vertex].compressed_edges.degree();

                // types::Neighbors neighbors = std::make_tuple(snapshoot.compressed_edges.get_edges(vertex), snapshoot.compressed_edges.degree(), true);
                return std::make_tuple(neighbors, weights, degrees, true);
            }

        private:
            pbbs::sequence<VertexEntry2> snapshot;
    };

    /**
     * @brief FlatGraphEntry represents one entry in the flat graph snapshot.
     */
    struct FlatGraphEntry2
    {
        types::Vertex* neighbors;
        types::Weight* weights;
        types::Degree degree;
        SamplerManager* samplers;

        /**
         * @brief FlatGraphEntry destructor.
         */
        ~FlatGraphEntry2()
        {
            pbbs::free_array(neighbors);
            pbbs::free_array(weights);
        }
    };

    /**
     * @brief FlatGraph2 stores for each vertex its neighbors, degree, weight and reference to its sampler manager.
     * This allows for O(1) access to the vertex, its edges and sampler manager and promises improved cache locality
     * at the expense of memory.
     */
    class FlatGraph2 : public Snapshot
    {
        public:
            /**
             * @brief FlatGraph constructor.
             *
             * @param vertices - number of vertices in a graph
             */
            explicit FlatGraph2(types::Vertex vertices)
            {
                this->snapshot = pbbs::sequence<FlatGraphEntry2>(vertices);
            }

            /**
            * @brief FlatGraph2 subscript operator overloading.
            *
            * @param vertex - graph vertex
            *
            * @return - snapshot entry for a given vertex
            */
            FlatGraphEntry2& operator[](types::Vertex vertex)
            {
                return this->snapshot[vertex];
            }

            /**
            * @brief FlatGraph size.
            *
            * @return - number of entries in a snapshot
            */
            types::Vertex size() final
            {
                return this->snapshot.size();
            }

            /**
            * @brief FlatGraph size in bytes. The memory footprint of a snapshot.
            *
            * @return - size of a snapshot in bytes
            */
            size_t size_in_bytes() final
            {
                size_t total = 0;
                std::vector<size_t> local_totals(this->snapshot.size());
                parallel_for(0, this->snapshot.size(), [&](auto index)
                {
                    local_totals[index] = sizeof(this->snapshot[index]) + (sizeof(this->snapshot[index].neighbors) * this->snapshot[index].degree);
                });

                #pragma omp critical
                for (size_t i = 0; i < local_totals.size(); ++i)
                {
                    total += local_totals[i];
                }

                return total;
            }

            /**
            * @brief Returns neighbors of a given vertex.
            *
            * @param vertex - graph vertex
            *
            * @return - neighbors of a given vertex
            */
            types::Neighbors neighbors(types::Vertex vertex) final
            {
                return std::make_tuple(snapshot[vertex].neighbors, snapshot[vertex].degree, false);
            }

            types::Neighbors2 neighbors2(types::Vertex vertex) final
            {
                return std::make_tuple(snapshot[vertex].neighbors, snapshot[vertex].weights, snapshot[vertex].degree, false);
            }
            
        private:
            pbbs::sequence<FlatGraphEntry2> snapshot;
    };
}

#endif
