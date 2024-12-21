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
#include <unordered_set>

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

            // 6. Init model
            // this->model = nullptr;
            
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

        void init_model(FlatGraph2& graph)
        {
            // switch (config::random_walk_model)
            // {
            //     case types::DEEPWALK:
            //         this->model = new DeepWalk(&graph);
            //         break;
            //     // case types::NODE2VEC:
            //     //     model = new Node2Vec(&graph, config::paramP, config::paramQ);
            //     //     break;
            //     default:
            //         std::cerr << "Unrecognized random walking model" << std::endl;
            //         std::exit(1);
            // }
            // return;
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
        [[nodiscard]] FlatVertexTree2 flatten_vertex_tree() const
        {

            types::Vertex n_vertices = this->number_of_vertices();
            auto flat_vertex_tree    = FlatVertexTree2(n_vertices);

            auto map_func = [&] (const Graph::E& entry, size_t ind)
            {
                const types::Vertex& key = entry.first;
                const auto& value = entry.second;
                flat_vertex_tree[key] = value;
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
#pragma optimize("", off) // 禁用优化
        void generate_initial_random_walks(RandomWalkModel& model)
        {
            auto graph             = this->flatten_graph();        // sequence of vertices snapshots
	        auto total_vertices    = this->number_of_vertices();
            auto walks_to_generate = total_vertices * config::walks_per_vertex;
            std::cout << "GENERATE " << walks_to_generate << " walks." << endl;
            auto cuckoo            = libcuckoo::cuckoohash_map<types::Vertex, std::vector<types::Vertex>>(total_vertices);   // walks hash which generate by pair fuction

            using VertexStruct  = std::pair<types::Vertex, VertexEntry2>;   // v_id -> compressed edges,compressed weights, compressed walks, and sampler manager
            auto vertices       = pbbs::sequence<VertexStruct>(total_vertices);  

            //RandomWalkModel* model;

            // 1、generate alias table
            if (config::biased_sampling) {
                std::cout << "Build alias table..." << std::endl;
                model.build_alias_table();
                std::cout << "Done." << std::endl;
            }

            // auto alias_table = this->model->get_alias_table();
            // for (size_t i = 0; i < alias_table.size(); i++)
            // {
            //     // auto current_src = vertices_to_resample[i];
            //     auto alias_table_i = alias_table[i];
            //     for (size_t j = 0; j < alias_table_i.size(); j++)
            //     {
            //         auto current_prob = alias_table_i[j].probability;
            //         auto current_dst = alias_table_i[j].second;
            //         std::cout << i << " " << current_dst << " " << current_prob << std::endl;
            //     }
            // }
            // 2、walk in parallel
            // parallel_for(0, walks_to_generate, [&] (types::WalkID walk_id)
            // {
            //     if (graph[walk_id % total_vertices].degree == 0) {
            //         types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + 0, walk_id % total_vertices});  
            //         cuckoo.insert(walk_id % total_vertices, std::vector<types::Vertex>());
            //         cuckoo.update_fn(walk_id % total_vertices, [&](auto& vector) {
            //             vector.push_back(hash);  
            //         });
            //         return;
            //     } 
                
            //     auto random = utility::Random(std::time(nullptr));               // By default random initialization
            //     if (config::deterministic_mode)
            //         random = utility::Random(walk_id / total_vertices);
            //     types::State state  = model.initial_state(walk_id % total_vertices);   // std::pair<Vertex, Vertex>

            //     for(types::Position position = 0; position < config::walk_length; position++) {
            //         if (!graph[state.first].samplers->contains(state.second))
			// 	        graph[state.first].samplers->insert(state.second , MetropolisHastingsSampler(state, &model)); 
                    
            //         // auto next_vertex = model.propose_vertex(state);
            //         // auto new_state = types::State(next_vertex, next_vertex);
            //         auto new_state = config::biased_sampling ? 
            //                         graph[state.first].samplers->find(state.second).sample(state, &model, true):
            //                         graph[state.first].samplers->find(state.second).sample(state, &model);

            //     std::cout << "walk_id: " << walk_id << " position: " << position << " state: " << state.first << "  sample" << new_state.first << std::endl;
            //     if (!cuckoo.contains(state.first))
			// 	  cuckoo.insert(state.first, std::vector<types::Vertex>());       
				
			//      types::PairedTriplet hash = (position != config::walk_length - 1) ?
			//                               pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + position, new_state.first}) :
			//                               pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + position, state.first}); // assign the current as next if EOW
            //     cuckoo.update_fn(state.first, [&](auto& vector) {
            //         vector.push_back(hash);        
            //     });

            //     // Assign the new state to the sampler
            //     state = new_state;
            //     } 
            // });

            
            // 2、walk in parallel
            for (types::WalkID walk_id = 0; walk_id < walks_to_generate; ++walk_id) {
                if (graph[walk_id % total_vertices].degree == 0) {
                    types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + 0, walk_id % total_vertices});  
                    cuckoo.insert(walk_id % total_vertices, std::vector<types::Vertex>());
                    cuckoo.update_fn(walk_id % total_vertices, [&](auto& vector) {
                        vector.push_back(hash);  
                    });
                    continue;
                }

                auto random = utility::Random(std::time(nullptr)); // By default random initialization
                if (config::deterministic_mode)
                    random = utility::Random(walk_id / total_vertices);
                
                types::State state = model.initial_state(walk_id % total_vertices); // std::pair<Vertex, Vertex>

                for (types::Position position = 0; position < config::walk_length; ++position) {
                    if (!graph[state.first].samplers->contains(state.second))
                        graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, &model));
                    
                    auto new_vert = model.biased_propose_vertex(state);

                    auto new_state = std::make_pair(new_vert, new_vert);
                    
                    // auto new_state = config::biased_sampling ? 
                    //                 graph[state.first].samplers->find(state.second).sample(state, &model, true) :
                    //                 graph[state.first].samplers->find(state.second).sample(state, &model);

                    // std::cout << "walk_id: " << walk_id << " position: " << position << " state: " << state.first << " sample: " << new_state.first << std::endl;

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
            }

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

            std::cout << "init walk squence: " << std::endl;
                for (auto i = 0; i < this->graph_tree.size(); i++) {
                std::cout << "src: " << i << ": ";
                std::cout << this->walk_simple_find(i) << std::endl;
            }
            return;
        }
#pragma optimize("", on) // 禁用优化


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
                    current_vertex = tree_node.value.compressed_walks.back().find_next_in_range(walk_id, position++, current_vertex);
                else
                    current_vertex = tree_node.value.compressed_walks.back().find_next(walk_id, position++, current_vertex); // operate on the front() as only one walk-tree exists after merging

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
                                                     RandomWalkModel& model,
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
        // v : src顶点
        // a : 旧值  b : 新值
        auto replace = [&, run_seq] (const intV& v, const VertexEntry2& a, const VertexEntry2& b)
        {
            // 合并edge_tree 和 weight_tree
            auto union_edge_tree = edge_plus::uniont(b.compressed_edges, a.compressed_edges, v, run_seq);  // 合并两个边树，构成新的c-tree 

            lists::deallocate(a.compressed_edges.plus);
            edge_plus::Tree_GC::decrement_recursive(a.compressed_edges.root, run_seq);                     // GC

            lists::deallocate(b.compressed_edges.plus);
            edge_plus::Tree_GC::decrement_recursive(b.compressed_edges.root, run_seq);					//  GC

            auto union_weight_tree = edge_plus::uniont(b.compressed_weights, a.compressed_weights, v, run_seq);

            lists::deallocate(a.compressed_weights.plus);
            edge_plus::Tree_GC::decrement_recursive(a.compressed_weights.root, run_seq);                     // GC

            lists::deallocate(b.compressed_weights.plus);
            edge_plus::Tree_GC::decrement_recursive(b.compressed_weights.root, run_seq);					//  GC
        

            auto triplets_to_delete_pbbs   = pbbs::new_array<std::vector<types::PairedTriplet>>(a.compressed_walks.size());   // 旧的walks，用于存储受影响的walk
		    // auto triplets_to_delete_vector = std::vector<std::vector<types::PairedTriplet>>();


            // compressed_walks 是一个vector,存放不同batch的walk树
            parallel_for(0, a.compressed_walks.size(), [&] (size_t index)
            {
                // 遍历所有的walk
                a.compressed_walks[index].iter_elms(v, [&](auto value)
                {
                    auto pair = pairings::Szudzik<types::PairedTriplet>::unpair(value);   // 解码walk,得到关键信息
                    auto walk_id = pair.first / config::walk_length;
                    auto position = pair.first - (walk_id * config::walk_length);
                    auto next = pair.second;

                    auto p_min_global = config::walk_length;
                    for (auto mav = a.compressed_walks[index].created_at_batch + 1; mav < batch_num; mav++)     // 遍历 mav2
                    {
                        if (MAVS2[mav].contains(walk_id))
                        {
                            auto temp_pos = get<0>((MAVS2[mav]).find(walk_id)); // it does not always contain this wid
                            if (temp_pos < p_min_global)
                                p_min_global = temp_pos; // TODO: an accumulated MAV with p_min up to that point might suffice
                        }
                    } 

                    // 如果pos 小于 p_min_global, 说明该walk需要重新采样
                    if (position < p_min_global) // TODO: this accepts all?
                    {
                        // take the triplet under consideration for the MAV and proceed normally
                        if (!rewalk_points.contains(walk_id))
                        {
                            rewalk_points.insert(walk_id, std::make_tuple(position, v, false));    
                        }
                        else
                        {
                            types::Position current_min_pos = get<0>(rewalk_points.find(walk_id));

                            if (current_min_pos > position)
                            {
                                rewalk_points.update(walk_id, std::make_tuple(position, v, false));   // 更细pos, 定位到第一个受影响的点
                            }
                        }
                        // std::cout << "walk_id: " << walk_id << " position: " << position << "need resample" << std::endl;
                    }
                    else {
                        triplets_to_delete_pbbs[index].push_back(value);  // pos大于p_min_global的triplet，需要删除，此时失效
                    }
                });
            });

		  // Create a new vector of compressed walks
		  // 用于删除的walk 序列
            vector<dygrl::CompressedWalks> vec_compwalks;
            for (auto j = 0; j < a.compressed_walks.size(); j++)
            {
                auto sequence = pbbs::sequence<types::Vertex>(triplets_to_delete_pbbs[j].size());
                parallel_for(0, triplets_to_delete_pbbs[j].size(), [&](auto k){
                    sequence[k] = triplets_to_delete_pbbs[j][k];                         // 二维转一维
                });
                pbbs::sample_sort_inplace(pbbs::make_range(sequence.begin(), sequence.end()), std::less<>());

                vec_compwalks.push_back(dygrl::CompressedWalks(sequence, v, 666, 666, batch_num - 1)); // dummy min,max, batch_num
            }

            // Do the differences
            // a.compressed_walks[ind] 删除 vec_compwalks[ind], 删除过时的walk
            std::vector<dygrl::CompressedWalks> new_compressed_vector;     // 删除后的新walk tree
            for (auto ind = 0; ind < a.compressed_walks.size(); ind++)
            {
                auto refined_walk_tree = walk_plus::difference(vec_compwalks[ind], a.compressed_walks[ind], v);   // return  a.compressed_walks[ind] / vec_compwalks[ind]
                new_compressed_vector.push_back(dygrl::CompressedWalks(refined_walk_tree.plus, refined_walk_tree.root, 666, 666, batch_num - 1)); // use dummy min, max, batch_num for now

                // deallocate the memory
                lists::deallocate(vec_compwalks[ind].plus);
                walk_plus::Tree_GC::decrement_recursive(vec_compwalks[ind].root);
                lists::deallocate(a.compressed_walks[ind].plus);
                walk_plus::Tree_GC::decrement_recursive(a.compressed_walks[ind].root);
            }

            // Merge all  the end trees into one 
            std::vector<dygrl::CompressedWalks> final_compressed_vector;
            final_compressed_vector.push_back(CompressedWalks(batch_num - 1));
            for (auto ind = 0; ind < new_compressed_vector.size(); ind++)
            {
                auto union_all_tree = walk_plus::uniont(new_compressed_vector[ind], final_compressed_vector[0], v);

                // deallocate the memory
                lists::deallocate(new_compressed_vector[ind].plus);
                walk_plus::Tree_GC::decrement_recursive(new_compressed_vector[ind].root); // deallocation is important for performance
                lists::deallocate(final_compressed_vector[0].plus);
                walk_plus::Tree_GC::decrement_recursive(final_compressed_vector[0].root);

                final_compressed_vector[0] = dygrl::CompressedWalks(union_all_tree.plus, union_all_tree.root, 666, 666, batch_num - 1); // refine the batch num after merging this node
            }
            return VertexEntry2(union_edge_tree, union_weight_tree, final_compressed_vector, b.sampler_manager);
        };


        // 合并c-tree
        
        this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, new_verts, num_starts, replace,true, run_seq);

        // 输出之前的walk 序列
        // std::cout << "before resample walk squece: " << std::endl;
        // for (auto i = 0; i < this->graph_tree.size(); i++) {
        //     std::cout << "src: " << i << ": ";
        //     std::cout << this->walk_simple_find(i) << std::endl;
        // }

        // 输出合并后的树
        // std::cout << "after merge : ------------------- " << std::endl;
        // for (auto i = 0; i < this->graph_tree.size(); i++)
        // {
        //     std::cout << "current src" << i << std::endl;
        //     auto tree_node = this->graph_tree.find(i);
        //     auto edges = tree_node.value.compressed_edges.get_edges(i);
        //     auto weights = tree_node.value.compressed_weights.get_edges(i);
        //     auto size = tree_node.value.compressed_edges.size();
        //     for (auto j = 0; j < size; j++) {
        //         std::cout << edges[j] << " " << weights[j] << " | ";
        //     }
        //     std::cout << std::endl;
        // }


        // Store/cache the MAV of each batch
		// 根据合并中的rewalk_points 来更新MAVS  WalkID, std::tuple<Position, Vertex, bool>
        std::cout << "rewalk_points in this batch : " << std::endl;
		for(auto& entry : rewalk_points.lock_table()) // todo: blocking?
		{
			MAVS2[batch_num].insert(entry.first, entry.second);         // 更新MAV 受影响的顶点  
            std::cout << entry.first << " : (" 
                    << std::get<0>(entry.second) << ", " 
                    << std::get<1>(entry.second) << ", " 
                    << std::get<2>(entry.second) << ")" 
                    << std::endl;
		}
		assert(rewalk_points.size() == MAVS2[batch_num].size());

        

		auto affected_walks = pbbs::sequence<types::WalkID>(rewalk_points.size());

        // todo: debug
		if (apply_walk_updates)
			this->batch_walk_update(rewalk_points, affected_walks, batch_num, model);        // 重新采样受影响的顶点   


        // 输出更新后的walk 序列
        // std::cout << "after rewalk walk squece: " << std::endl;
        // for (auto i = 0; i < this->graph_tree.size(); i++) {
        //     std::cout << "src: " << i << ": ";
        //     std::cout  << this->walk_simple_find(i) << std::endl;
        // }

        return affected_walks;
    }

	/**
	 * @brief Updates affected walks in batch mode.
	 * @param types::MapOfChanges - rewalking points: walk_id --> tuple(position, v, should_reset)
	 */
	// 重新采样受影响的顶点
	void batch_walk_update(types::MapAffectedVertices& rewalk_points, pbbs::sequence<types::WalkID>& affected_walks, int batch_num, RandomWalkModel& model)
	{
		types::ChangeAccumulator inserts = types::ChangeAccumulator();        // 插入累加器

		uintV index = 0;

		for(auto& entry : rewalk_points.lock_table()) // todo: blocking?
		{
			affected_walks[index++] = entry.first;      // affected_walks, 存储需要重新游走的walk id
		}

	    auto graph = this->flatten_graph();           // 获取快照

        // 重新生成采样空间
        auto vertCuckoo = libcuckoo::cuckoohash_map<size_t, bool>();
        parallel_for(0, affected_walks.size(), [&](auto index)
        {
            auto current_src = affected_walks[index] % this->number_of_vertices();
            vertCuckoo.insert(current_src, true);
        });
        auto affected_vertices = vertCuckoo.size();
        vector<size_t> vertices_to_resample(affected_vertices);
        size_t curIndex = 0;
        for (auto& entry : vertCuckoo.lock_table())
        {
            vertices_to_resample[curIndex++] = entry.first;
        }
        assert(curIndex == affected_vertices);

        // 并行重新生成采样空间
        // parallel_for(0, affected_vertices, [&](auto index)
        // {
        //     //model->build_alias_table();
        //     this->model->build_alias_table_single(vertices_to_resample[index]);
        // });

        // TODO: 无法捕获
        //输出受影响的顶点
        // std::cout << "affected vertices: " << affected_vertices << std::endl;
        // for (size_t i = 0; i < affected_vertices; i++)
        // {
        //     auto current_src = vertices_to_resample[i];
        //     std::cout << current_src << " ";
        // }
        // std::cout <<std::endl<< "before rebuild alias table ------" << std::endl;

        // auto alias_table = model.get_alias_table();
        // for (size_t i = 0; i < alias_table.size(); i++)
        // {
        //     auto alias_table_i = alias_table[i];
        //     for (size_t j = 0; j < alias_table_i.size(); j++)
        //     {
        //         auto current_prob = alias_table_i[j].probability;
        //         auto current_dst = alias_table_i[j].second;
        //         std::cout << i << " " << current_dst << " " << current_prob << std::endl;
        //     }
        // }

        // 重新生成采样空间
        // todo 更新model snapshot
        auto new_graph_flat = this->flatten_graph();

        DeepWalk& derived_model = dynamic_cast<DeepWalk&>(model); // 强制类型转换
        derived_model.update_snapshot(&new_graph_flat);     
        model = derived_model;          // 更新model snapshot, TODO: bugfix gc

        //derived_model.update_snapshot(&new_graph_flat);

        for (auto& entry : rewalk_points.lock_table())
		{
            model.build_alias_table_single(entry.first);
		}

        // std::cout << "after rebuild alias table" << std::endl;
        // auto alias_table = model.get_alias_table();
        // for (size_t i = 0; i < alias_table.size(); i++)
        // {
        //     auto alias_table_i = alias_table[i];
        //     for (size_t j = 0; j < alias_table_i.size(); j++)
        //     {
        //         auto current_prob = alias_table_i[j].probability;
        //         auto current_dst = alias_table_i[j].second;
        //         std::cout << i << " " << current_dst << " " << current_prob << std::endl;
        //     }
        // }

        parallel_for(0, affected_walks.size(), [&](auto index)
        {
            auto entry = rewalk_points.template find(affected_walks[index]);   // 获取需要重新采样的vertex

            auto current_position        = std::get<0>(entry);
            auto current_vertex_old_walk = std::get<1>(entry);
            auto should_reset            = std::get<2>(entry);

            auto current_vertex_new_walk = current_vertex_old_walk;

            if (should_reset) // todo: clear out if this is needed
            {   
                current_position = 0;                        // 直接从头采样，相当于不使用增量更新
                current_vertex_new_walk = current_vertex_old_walk = affected_walks[index] % this->number_of_vertices();
            }

            if (graph[current_vertex_new_walk].degree == 0)       // 如果度为零，此时不需要再更新下去
            {
                types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({affected_walks[index] * config::walk_length + current_position, current_vertex_new_walk});    // walk 编码， 下一跳为current_vertex_new_walk
                if (!inserts.contains(current_vertex_new_walk)) { 
                    inserts.insert(current_vertex_new_walk, std::vector<types::PairedTriplet>());   // 插入累加器初始化
                }                
                inserts.update_fn(current_vertex_new_walk, [&](auto& vector) {
                    vector.push_back(hash);                                                        // vecotr存入所有待更新的walk
                });

                return;
            }

            auto random = utility::Random(std::time(nullptr));
            if (config::deterministic_mode)
                random = utility::Random(affected_walks[index] / this->number_of_vertices());

            auto state = model.initial_state(current_vertex_new_walk);

            // 重新采样
            for (types::Position position = current_position; position < config::walk_length; position++)
            {
                if (!graph[state.first].samplers->contains(state.second))
                    graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, &model));

                auto temp_state = config::biased_sampling ? 
                                    graph[state.first].samplers->find(state.second).sample(state, &model, true):
                                    graph[state.first].samplers->find(state.second).sample(state, &model);

                // if (state.first == 4) {
                //     std::cout << "state: " << state.first << " " << state.second << std::endl;
                //     std::cout << "temp_state: " << temp_state.first << " " << temp_state.second << std::endl;
                //     std::cout << "position: " << position << std::endl;
                //     std::cout << "current_vertex_new_walk: " << current_vertex_new_walk << std::endl;
                // }

                if (config::deterministic_mode)
                {
                    state = model.new_state(state, graph[state.first].neighbors[random.irand(graph[state.first].degree)]);
                }
                else
                    state = temp_state;

                //number_of_sampled_vertices++;
                types::PairedTriplet hash = (position != config::walk_length - 1) ?
                                            pairings::Szudzik<types::Vertex>::pair({affected_walks[index] * config::walk_length + position, state.first}) : // new sampled next
                                            pairings::Szudzik<types::Vertex>::pair({affected_walks[index] * config::walk_length + position, current_vertex_new_walk});  // 最后一跳

                if (!inserts.contains(current_vertex_new_walk)) {
                    inserts.insert(current_vertex_new_walk, std::vector<types::PairedTriplet>());
                }
                inserts.update_fn(current_vertex_new_walk, [&](auto& vector) {
                    vector.push_back(hash);                                // 插入累加器加入hash
                });

                // std::cout << "walk " << affected_walks[index] << " " <<" position " << position << "  sample new vertex " << state.first << std::endl;
                // Then, change the current vertex in the new walk
                current_vertex_new_walk = state.first;
            }

        });

        // 合并walk 数据
		using VertexStruct  = std::pair<types::Vertex, VertexEntry2>;
		auto insert_walks  = pbbs::sequence<VertexStruct>(inserts.size());

		auto temp_inserts = pbbs::sequence<std::pair<types::Vertex, std::vector<types::PairedTriplet>>>(inserts.size());
		auto skatindex = 0;
		for (auto& item: inserts.lock_table()) // TODO: This cannot be in parallel. Cuckoo-hashmap limitation
		{
			temp_inserts[skatindex++] = std::make_pair(item.first, item.second);
		}

		parallel_for(0, temp_inserts.size(), [&](auto j){
		  auto sequence = pbbs::sequence<types::Vertex>(temp_inserts[j].second.size());

		  parallel_for(0, temp_inserts[j].second.size(), [&](auto i){
			sequence[i] = temp_inserts[j].second[i];
		  });

		  pbbs::sample_sort_inplace(pbbs::make_range(sequence.begin(), sequence.end()), std::less<>());
		  vector<dygrl::CompressedWalks> vec_compwalks;
		  vec_compwalks.push_back(dygrl::CompressedWalks(sequence, temp_inserts[j].first, 666, 666, batch_num));
		  insert_walks[j] = std::make_pair(temp_inserts[j].first, VertexEntry2(types::CompressedEdges(), types::CompressedEdges(), vec_compwalks, new dygrl::SamplerManager(0)));
		});


		pbbs::sample_sort_inplace(pbbs::make_range(insert_walks.begin(), insert_walks.end()), [&](auto& x, auto& y) {
		  return x.first < y.first;
		});

		// TODO: this when simple merge
        // 保留x but replace y
		auto replaceI = [&] (const uintV& src, const VertexEntry2& x, const VertexEntry2& y)
		{
		  auto x_prime = x.compressed_walks;
		  if (y.compressed_walks.back().size() != 0)
			  x_prime.push_back(y.compressed_walks.back()); // y has only one walk tree
		  return VertexEntry2(x.compressed_edges, x.compressed_weights, x_prime, x.sampler_manager);
		};

		cout << "\n(insert) -- For batch-" << batch_num << " we are touching " << insert_walks.size() << " / " << number_of_vertices() << " vertices" << endl;
		this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, insert_walks.begin(), insert_walks.size(), replaceI, true);


        if (batch_num % config::merge_frequency == 0)
        {
            cout << "\n------ merge at batch---------" << batch_num << endl;
            // TODO: 合并walk数据
            // TODO: bugfix 无法合并tree
            merge_walk_trees_all_vertices_parallel(batch_num); // ok merge all old nodes

            std::cout << "merge_walk_trees_all_vertices_parallel" << std::endl;
            std::cout << "current walk squece: " << std::endl;
                for (auto i = 0; i < this->graph_tree.size(); i++) {
                std::cout << "src: " << i << ": ";
                std::cout << this->walk_simple_find(i) << std::endl;
            }
        }
    
    }



	/**
	 * @brief Merges the walk-trees of each vertex in the hybrid-tree such that in the end each vertex has only one walk-tree
	 */
	// 合并walk tree
    void merge_walk_trees_all_vertices_parallel(int num_batches_so_far)
    {
        libcuckoo::cuckoohash_map<types::Vertex, std::vector<std::vector<types::PairedTriplet>>> all_to_delete; // let's use a vector

        auto flat_graph = this->flatten_vertex_tree();      // 获取快照
        parallel_for(0, this->number_of_vertices(), [&](size_t i)
        {
            int inc = 0;

            auto triplets_to_delete_pbbs   = pbbs::new_array<std::vector<types::PairedTriplet>>(flat_graph[i].compressed_walks.size());
            auto triplets_to_delete_vector = std::vector<std::vector<types::PairedTriplet>>();

            // traverse each walk-tree and find out the obsolete triplets and create corresponding "deletion" walk-trees
            // 遍历所有的walk-tree，删除其中过时的walk数目
            for (auto wt = flat_graph[i].compressed_walks.begin(); wt != flat_graph[i].compressed_walks.end(); wt++) // TODO: make this parallel for. REMARK: does not pay off
            {
                // Define the triplets to delete vector for each walk-tree
                wt->iter_elms(i, [&](auto enc_triplet)
                {
                    auto pair = pairings::Szudzik<types::Vertex>::unpair(enc_triplet);     // 获取walk

                    auto walk_id  = pair.first / config::walk_length;
                    auto position = pair.first - (walk_id * config::walk_length);
                    auto next_vertex   = pair.second;

                    auto p_min_global = config::walk_length;
                    for (auto mav = wt->created_at_batch+1; mav < num_batches_so_far + 1; mav++) // CAUTION: #b in the input + 1
                    {
                        if (MAVS2[mav].template contains(walk_id))
                        {
                            auto temp_pos = get<0>((MAVS2[mav]).template find(walk_id)); // it does not always contain this wid
                            if (temp_pos < p_min_global)
                                p_min_global = temp_pos; // TODO: an accumulated MAV with p_min up to that point might suffice
                        }
                    } 
                    // Check the relationship of the triplet with respect to the p_min_global or the w
                    if (position < p_min_global) // TODO: this accepts all?
                    {
                        ; // the triplet is still valid so it stays
                    }
                    else
                    {
                        triplets_to_delete_pbbs[inc].push_back(enc_triplet);
                    }

                });

                triplets_to_delete_vector.push_back(triplets_to_delete_pbbs[inc]);
                inc++;
        }

        // add the triplets to delete for this vertex in a hashmap
        //  all_to_delete 也是一个map  vertex -> triplets 的 集合
        if (!all_to_delete.contains(i))
                all_to_delete.insert(i, std::vector<std::vector<types::PairedTriplet>>());
            all_to_delete.update_fn(i, [&](auto& vector) {
                vector = triplets_to_delete_vector;
            });
        });

        auto temp_deletes = pbbs::sequence<std::pair<types::Vertex, std::vector<std::vector<types::PairedTriplet>>>>(all_to_delete.size());
        auto skatindex = 0;

        for (auto& item: all_to_delete.lock_table()) // TODO: This cannot be in parallel. Cuckoo-hashmap limitation
        {
            temp_deletes[skatindex++] = std::make_pair(item.first, item.second);
        }

        // 将 all_to_delete  转换为VertexEntry2
        using VertexStruct = std::pair<types::Vertex, VertexEntry2>;   
        auto delete_walks  = pbbs::sequence<VertexStruct>(all_to_delete.size());
        cout << "all_to_delete size: " << all_to_delete.size() << endl;
        parallel_for(0, temp_deletes.size(), [&](auto kkk)
        {
            vector<dygrl::CompressedWalks> vec_compwalks;

            auto vertex_id = temp_deletes[kkk].first;

            for (auto j = 0; j < temp_deletes[kkk].second.size(); j++)
            {
                auto sequence = pbbs::sequence<types::Vertex>(temp_deletes[kkk].second[j].size());
                parallel_for(0, temp_deletes[kkk].second[j].size(), [&](auto k){
                    sequence[k] = temp_deletes[kkk].second[j][k];
                });
                pbbs::sample_sort_inplace(pbbs::make_range(sequence.begin(), sequence.end()), std::less<>());

                vec_compwalks.push_back(dygrl::CompressedWalks(sequence, temp_deletes[kkk].first, 666, 666, num_batches_so_far)); // dummy min,max, batch_num
            }

            // 都是需要删除的 walk-tree
            delete_walks[kkk] = std::make_pair(temp_deletes[kkk].first, VertexEntry2(types::CompressedEdges(),types::CompressedEdges(), vec_compwalks, new dygrl::SamplerManager(0)));
        });

            // Sort the delete walks
            pbbs::sample_sort_inplace(pbbs::make_range(delete_walks.begin(), delete_walks.end()), [&](auto& x, auto& y) {
                return x.first < y.first;
            });

            auto replaceI = [&] (const uintV& src, const VertexEntry2& x, const VertexEntry2& y)
            {
                assert(x.compressed_walks.size() == y.compressed_walks.size());
                std::vector<dygrl::CompressedWalks> new_compressed_vector;
                for (auto ind = 0; ind < x.compressed_walks.size(); ind++)
                {
                    // return x / y
                    auto refined_walk_tree = walk_plus::difference(y.compressed_walks[ind], x.compressed_walks[ind], src);
                    new_compressed_vector.push_back(dygrl::CompressedWalks(refined_walk_tree.plus, refined_walk_tree.root, 666, 666, num_batches_so_far)); // use dummy min, max, batch_num for now

                    // deallocate the memory
                    lists::deallocate(x.compressed_walks[ind].plus);
                    walk_plus::Tree_GC::decrement_recursive(x.compressed_walks[ind].root);
                    lists::deallocate(y.compressed_walks[ind].plus);
                    walk_plus::Tree_GC::decrement_recursive(y.compressed_walks[ind].root);
                }

                // merge the refined walk-trees here
                std::vector<dygrl::CompressedWalks> final_compressed_vector;

                final_compressed_vector.push_back(CompressedWalks(num_batches_so_far));
                for (auto ind = 0; ind < new_compressed_vector.size(); ind++)
                {
                    auto union_all_tree = walk_plus::uniont(new_compressed_vector[ind], final_compressed_vector[0], src);

                    // deallocate the memory
                    lists::deallocate(new_compressed_vector[ind].plus);
                    walk_plus::Tree_GC::decrement_recursive(new_compressed_vector[ind].root);
                    lists::deallocate(final_compressed_vector[0].plus);
                    walk_plus::Tree_GC::decrement_recursive(final_compressed_vector[0].root);

                    final_compressed_vector[0] = dygrl::CompressedWalks(union_all_tree.plus, union_all_tree.root, 666, 666, num_batches_so_far);
            }

            return VertexEntry2(x.compressed_edges, x.compressed_weights, final_compressed_vector, x.sampler_manager);
        };
        this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, delete_walks.begin(), delete_walks.size(), replaceI, true);
        return;
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