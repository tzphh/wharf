#ifndef WEIGHT_H
#define WEIGHT_H

#include "common/types.h"
#include "config.h"

namespace weight{
    /**
     * @brief Get the weight of an encoded value.
     * 
     * @param value : tree node value
     */
    inline uintW get_weight(uintW value) {
        return (value % config::graph_vertices);
    }

    /**
     * @brief Get the dst of an encoded value.
     */
    inline uintV get_dst(uintW value) {
        return (value / config::graph_vertices);
    }

}

#endif