#ifndef COMPRESSED_WEIGHTS_H
#define COMPRESSED_WEIGHTS_H

#include <graph/tree_plus/edge_plus.h>
#include <graph/tree_plus/walk_plus.h>

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
    * CompressedWalks is a structure that stores graph random walks in a compressed format.
    * It essentially represents a compressed purely functional tree (C-Tree) that achieves compression
    * based on differential coding.
    */
    class CompressedWeights : public walk_plus::treeplus
    {
        public:
			int created_at_batch;

            /**
             * @brief CompressedWeights default constructor.
             */
            CompressedWeights() : walk_plus::treeplus(), created_at_batch(0) {};
            CompressedWeights(int batch_num) : walk_plus::treeplus(), created_at_batch(batch_num) {};
    };
}

#endif
