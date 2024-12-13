#ifndef STGRAPH_H
#define STGRAPH_H

#include <graph/api.h>
#include <cuckoohash_map.hh>
#include <pbbslib/utilities.h>

#include <config.h>
#include <pairings.h>
#include <vertex.h>
#include <snapshot2.h>
#include <utility.h>

#include <models/deepwalk.h>
#include <models/node2vec.h>
#include <set>
#include <memory>

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


            // 5. Initialize the mav
            for (size_t i = 0; i < config::max_batch_num; i++) {
                this->MAVS2.push_back(types::MapAffectedVertices());
            }

            // 5. GC
            if (free_memory)
            {
                pbbs::free_array(offsets);
                pbbs::free_array(weights);
                pbbs::free_array(edges);
            }
		    vertices.clear();
        }

        /**
         * @brief Number of vertices in a graph.
         * 
         * @return - the number of vertices in a graph
         */
        [[nodiscard]] auto number_of_vertices() const    
        {
            size_t n = this->graph_tree.size();
            auto last_vertex = this->graph_tree.select(n - 1);

            return n > 0 ? last_vertex.value.first + 1 : 0;
        }

        /**
         * @brief Number of edges in a graph.
         *
         * @return - the number of edges in a graph
         */
        [[nodiscard]] auto number_of_edges() const
        {
            return this->graph_tree.aug_val();            // aug_t from_entry
        }

        /**
         * @brief Traverses vertices and applies mapping function.
         *
         * @tparam F
         *
         * @param map_f   - map function
         * @param run_seq - determines whether to run part of the code sequantially
         * @param granularity
         */
        template<class Function>
        void map_vertices(Function map_function, bool run_seq = false, size_t granularity = utils::node_limit) const
        {
            this->graph_tree.map_elms(map_function, run_seq, granularity);
        }

        /**
         * @brief Flattens the vertex tree to an array of vertex entries.
         *
         * @return - the sequence of pointers to graph vertex entries
         */
        [[nodiscard]] std::unique_ptr<FlatVertexTree2> flatten_vertex_tree() const
        {
            types::Vertex n_vertices = this->number_of_vertices();
            auto flat_vertex_tree = std::make_unique<FlatVertexTree2>(n_vertices);

            auto map_func = [&] (const Graph::E& entry, size_t ind)
            {
                const types::Vertex& key = entry.first;
                const auto& value = entry.second;
                (*flat_vertex_tree)[key] = value;
            };

            this->map_vertices(map_func);
            return flat_vertex_tree;
        }
    
        /**
        * @brief Flattens the graph to an array of vertices, their degrees, neighbors, weights and sampler managers.
        *
        * @return - the sequence of vertices, their degrees, neighbors, weights and sampler managers
        */
        [[nodiscard]] FlatGraph2 flatten_graph() const
        {
            size_t n_vertices = this->number_of_vertices();
            auto flat_graph   = FlatGraph2(n_vertices);

            auto map_func = [&] (const Graph::E& entry, size_t ind)
            {
                const uintV& key  = entry.first;
                const auto& value = entry.second;

                flat_graph[key].neighbors = entry.second.compressed_edges.get_edges(key);
                flat_graph[key].weights   = reinterpret_cast<types::Weight*>(entry.second.compressed_weights.get_edges(key));
                flat_graph[key].degree    = entry.second.compressed_edges.degree();
                flat_graph[key].samplers  = entry.second.sampler_manager;
            };

            this->map_vertices(map_func);

            return flat_graph;
        }

        /**
         * @brief Destroys malin instance.
         */
        void destroy()
        {
            this->graph_tree.~Graph();
            this->graph_tree.root = nullptr;
        }

        /**
        * @brief Creates initial set of random walks. 
        * @TODO: generate walks according to weights.  
        */
        void generate_initial_random_walks()
        {
            auto graph             = this->flatten_graph();        // sequence of vertices snapshots
	        auto total_vertices    = this->number_of_vertices();
            auto walks_to_generate = total_vertices * config::walks_per_vertex;
            std::cout << "GENERATE " << walks_to_generate << " walks." << endl;
            auto cuckoo            = libcuckoo::cuckoohash_map<types::Vertex, std::vector<types::Vertex>>(total_vertices);   // walks hash which generate by pair fuction

            using VertexStruct  = std::pair<types::Vertex, VertexEntry2>;   // v_id -> compressed edges,compressed weights, compressed walks, and sampler manager
            auto vertices       = pbbs::sequence<VertexStruct>(total_vertices);  

            RandomWalkModel* model;
            switch (config::random_walk_model)
            {
                case types::DEEPWALK:
                    model = new DeepWalk(&graph);
                    break;
                // case types::NODE2VEC:
                //     model = new Node2Vec(&graph, config::paramP, config::paramQ);
                //     break;
                default:
                    std::cerr << "Unrecognized random walking model" << std::endl;
                    std::exit(1);
            }

            // 1、generate alias table
            if (config::biased_sampling) {
                std::cout << "Build alias table..." << std::endl;
                model->build_alias_table();
                std::cout << "Done." << std::endl;
            }
            // 2、walk in parallel
            parallel_for(0, walks_to_generate, [&] (types::WalkID walk_id)
            {
                if (graph[walk_id % total_vertices].degree == 0) {
                    types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + 0, walk_id % total_vertices});  
                    cuckoo.insert(walk_id % total_vertices, std::vector<types::Vertex>());
                    cuckoo.update_fn(walk_id % total_vertices, [&](auto& vector) {
                        vector.push_back(hash);  
                    });
                    return;
                }
                
                auto random = utility::Random(std::time(nullptr));               // By default random initialization
                if (config::deterministic_mode)
                    random = utility::Random(walk_id / total_vertices);
                types::State state  = model->initial_state(walk_id % total_vertices);   // std::pair<Vertex, Vertex>

                for(types::Position position = 0; position < config::walk_length; position++) {
                    if (!graph[state.first].samplers->contains(state.second))
				        graph[state.first].samplers->insert(state.second , MetropolisHastingsSampler(state, model)); 
                    auto new_state = config::biased_sampling ? 
                                    graph[state.first].samplers->find(state.second).sample(state, model, true):
                                    graph[state.first].samplers->find(state.second).sample(state, model);

                if (!cuckoo.contains(state.first))
				  cuckoo.insert(state.first, std::vector<types::Vertex>());       
				
			     types::PairedTriplet hash = (position != config::walk_length - 1) ?
			                              pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + position, new_state.first}) :
			                              pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + position, state.first}); // assign the current as next if EOW
                cuckoo.update_fn(state.first, [&](auto& vector) {
                    vector.push_back(hash);        
                });

                // Assign the new state to the sampler
                state = new_state;
                } 
            });

		    // 3. build vertices
            parallel_for(0, total_vertices, [&](types::Vertex vertex)
            {
                if (cuckoo.contains(vertex))            // vertex has walks
                {
                    auto triplets = cuckoo.find(vertex);
                    auto sequence = pbbs::sequence<types::Vertex>(triplets.size());

                    parallel_for(0, triplets.size(), [&](size_t index)
                    {
                        sequence[index] = triplets[index];
                    });

                    // assert walks are created at batch 0 and sorted by vertex id
                    pbbs::sample_sort_inplace(pbbs::make_range(sequence.begin(), sequence.end()), std::less<>()); 
                    vector<dygrl::CompressedWalks> vec_compwalks;
                    vec_compwalks.push_back(dygrl::CompressedWalks(sequence, vertex, 666, 666, /*next_min[vertex], next_max[vertex],*/ 0)); // this is created at time 0
                    vertices[vertex] = std::make_pair(vertex, VertexEntry2(types::CompressedEdges(), types::CompressedEdges(), vec_compwalks, new dygrl::SamplerManager(0)));
                }
                else
                {
                    vector<dygrl::CompressedWalks> vec_compwalks; 
                    vec_compwalks.push_back(dygrl::CompressedWalks(0)); // at batch 0
                    vertices[vertex] = std::make_pair(vertex, VertexEntry2(types::CompressedEdges(), types::CompressedEdges(), vec_compwalks,new dygrl::SamplerManager(0)));  
                }
            });

            // 4. Merge walks, walks are sorted by vector
            auto replace = [&] (const uintV& src, const VertexEntry2& x, const VertexEntry2& y)   
            {
                auto x_prime = x;
                x_prime.compressed_walks.push_back(y.compressed_walks.back()); // y has only one walk tree

                return x_prime;
            };
            this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, vertices.begin(), vertices.size(), replace, true);
            delete model;

            return;
        }

        /**
         * @brief Walks through the walk given walk id.
         *
         * @param walk_id - unique walk ID
         *
         * @return - walk string representation
         */
        std::string walk_simple_find(types::WalkID walk_id)
        {
            // 1. Grab the first vertex in the walk
            types::Vertex current_vertex = walk_id % this->number_of_vertices();
            std::stringstream string_stream;
            types::Position position = 0;

            // 2. Walk
            types::Vertex previous_vertex;
            while (true)
            {
                string_stream << current_vertex << " ";
                auto tree_node = this->graph_tree.find(current_vertex);

                // Cache the previous current vertex before going to the next
                previous_vertex = current_vertex;

                // uses only simple find_next
                current_vertex = tree_node.value.compressed_walks.front().find_next(walk_id, position++, current_vertex); // operate on the front() as only one walk-tree exists after merging
                if (current_vertex == previous_vertex)     
                    break;
            }

            return string_stream.str();
        }

        /**
         * @brief Walks through the walk given walk id.
         *
         * @param walk_id - unique walk ID
         *
         * @return - walk string representation
         */
        std::string walk_silent(types::WalkID walk_id)
        {
            // 1. Grab the first vertex in the walk
            types::Vertex current_vertex = walk_id % this->number_of_vertices();
            std::stringstream string_stream;
            types::Position position = 0;

            // 2. Walk
            types::Vertex previous_vertex;
            while (true)
            {
                string_stream << current_vertex << " ";

                auto tree_node = this->graph_tree.find(current_vertex);

                // Cache the previous current vertex before going to the next
                previous_vertex = current_vertex;

                // uses only simple find_next
                if (config::range_search_mode)
                    current_vertex = tree_node.value.compressed_walks.front().find_next_in_range(walk_id, position++, current_vertex);
                else
                    current_vertex = tree_node.value.compressed_walks.front().find_next(walk_id, position++, current_vertex); // operate on the front() as only one walk-tree exists after merging

                if (current_vertex == previous_vertex)
                    break;
            }

            return string_stream.str();
        }

    // TODO : add support for batch update

    	/**
	* @brief Inserts a batch of edges in the graph.
	*
	* @param m                  - size of the batch
	* @param edges              - atch of edges to insert
    * @param weights             - edge weights associated with the edges
	* @param batch_num          - batch number, indicates the batch number
	* @param sorted             - sort the edges in the batch
	* @param remove_dups        - removes duplicate edges in the batch
	* @param nn				    - limit the number of neighbors to be considered
	* @param apply_walk_updates - decides if walk updates will be executed
	*/
	pbbs::sequence<types::WalkID> insert_edges_batch(size_t m,
	                                                 std::tuple<uintV, uintV>* edges,
                                                     uintW* weights,
	                                                 int batch_num,
	                                                 bool sorted = false,
	                                                 bool remove_dups = false,
	                                                 size_t nn = std::numeric_limits<size_t>::max(),
	                                                 bool apply_walk_updates = true,
	                                                 bool run_seq = false)
	{
        auto fl = run_seq ? pbbs::fl_sequential : pbbs::no_flag;

        // 1. Set up , make sure edges and weights are sorted and decoupled
        using Edge = std::tuple<uintV, uintV>;      
        auto E = pbbs::make_range(edges, edges + m);
        auto W = pbbs::make_range(weights, weights + m);

		// 2. Pack the starts vertices of edges
		auto start_im = pbbs::delayed_seq<size_t>(m, [&] (size_t i)
		{
		  return (i == 0 || (get<0>(E[i]) != get<0>(E[i - 1])));
		});
		auto starts = pbbs::pack_index<size_t>(start_im, fl);   
		size_t num_starts = starts.size();

        // 3. Build new vertices to insert and pack edges to new_verts
        using KV = std::pair<uintV, VertexEntry2>;
        KV* new_verts = pbbs::new_array<KV>(num_starts);
        parallel_for(0, num_starts, [&] (size_t i) {
            size_t off = starts[i];                                                                   // offset
            size_t deg = (i == num_starts - 1) ? m - off : starts[i + 1] - off;                      // degree
            uintV v = get<0>(E[off]);                                                                // source vertex
            auto SE = pbbs::delayed_seq<uintV>(deg, [&] (size_t i) { return get<1>(E[off + i]); });   // get dst
            auto SW = pbbs::delayed_seq<uintV>(deg, [&] (size_t i) { return W[off + i]; });   // get weight

            vector<dygrl::CompressedWalks> vec_compwalks; 
		    vec_compwalks.push_back(dygrl::CompressedWalks(batch_num));        // 初始化batch_num

            new_verts[i] = make_pair(v, VertexEntry2(types::CompressedEdges(SE, v, fl),
                                                     types::CompressedEdges(SW, v, fl),
                                                     vec_compwalks,
                                                     new dygrl::SamplerManager(0)));
        });
        types::MapAffectedVertices rewalk_points = types::MapAffectedVertices();  // MAV 受影响的顶点的映射
        // 定义了一个 replace Lambda 函数, 比较关键 


    }

    private:
        Graph graph_tree;   // same name as in Wharf
        std::vector<types::MapAffectedVertices> MAVS2;    // MAV 受影响的顶点
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