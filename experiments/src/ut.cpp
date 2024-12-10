#include <stgraph.h>

void get_walkpath(commandLine& command_line)
{
    string fname       = string(command_line.getOptionValue("-f", default_file_name));
    bool mmap          = command_line.getOption("-m");
    bool is_symmetric  = command_line.getOption("-s");
    bool compressed    = command_line.getOption("-c");
    size_t n_parts     = command_line.getOptionLongValue("-nparts", 1);

    size_t w           = command_line.getOptionLongValue("-w", config::walks_per_vertex);
    size_t l           = command_line.getOptionLongValue("-l", config::walk_length);
    string model       = string(command_line.getOptionValue("-model", "deepwalk"));
    double p           = command_line.getOptionDoubleValue("-p", config::paramP);
    double q           = command_line.getOptionDoubleValue("-q", config::paramQ);
    string init        = string(command_line.getOptionValue("-init", "weight"));

    string det         = string(command_line.getOptionValue("-det", "false"));
    string rs          = string(command_line.getOptionValue("-rs", "true"));
	size_t num_of_batches   = command_line.getOptionIntValue("-nb", 10);
	size_t half_of_bsize    = command_line.getOptionIntValue("-bs", 5000);
	size_t merge_freq       = command_line.getOptionIntValue("-mergefreq", 1);
	config::merge_frequency = merge_freq;
	string merge_mode       = string(command_line.getOptionValue("-mergemode", "parallel"));
	if (merge_mode == "parallel")
	{
		config::parallel_merge_wu = true;
		std::cout << "Parallel Merge and WU" << std::endl;
	}
	else
	{
		config::parallel_merge_wu = false;
		std::cout << "Serial Merge and WU" << std::endl;
	}

    config::walks_per_vertex = w;
    config::walk_length      = l;

    std::cout << "Walks per vertex: " << (int) config::walks_per_vertex << std::endl;
    std::cout << "Walk length: " << (int) config::walk_length << std::endl;

    if (model == "deepwalk")
    {
        config::random_walk_model = types::RandomWalkModelType::DEEPWALK;

        std::cout << "Walking model: DEEPWALK" << std::endl;
    }
    else if (model == "node2vec")
    {
        config::random_walk_model = types::RandomWalkModelType::NODE2VEC;
        config::paramP = p;
        config::paramQ = q;

        std::cout << "Walking model: NODE2VEC | Params (p,q) = " << "(" << config::paramP << "," << config::paramQ << ")" << std::endl;
    }
    else
    {
        std::cerr << "Unrecognized walking model! Abort" << std::endl;
        std::exit(1);
    }

    if (init == "burnin")
    {
        config::sampler_init_strategy = types::SamplerInitStartegy::BURNIN;

        std::cout << "Sampler strategy: BURNIN" << std::endl;
    }
    else if (init == "weight")
    {
        config::sampler_init_strategy = types::SamplerInitStartegy::WEIGHT;

        std::cout << "Sampler strategy: WEIGHT" << std::endl;
    }
    else if (init == "random")
    {
        config::sampler_init_strategy = types::SamplerInitStartegy::RANDOM;

        std::cout << "Sampler strategy: RANDOM" << std::endl;
    }
    else
    {
        std::cerr << "Unrecognized sampler init strategy" << std::endl;
        std::exit(1);
    }

    // Set up the range search mode ------
    if (rs == "true")
        config::range_search_mode = true;
    else
        config::range_search_mode = false;

    // Set up the deterministic mode
    if (det == "true")
        config::deterministic_mode = true;
    else
        config::deterministic_mode = false;
    // ------------------------------------

    size_t n;
    size_t m;
    uintE* offsets;
    uintV* edges;
    uintW* weights;
    std::tie(n, m, offsets, edges, weights) = read_weighted_graph(fname.c_str(), is_symmetric, mmap);
    for (size_t i = 0; i < n; i++)
    {
        std::cout << "Vertex " << i << " has " << offsets[i + 1] - offsets[i] << " neighbors" << std::endl;
    }
    for (size_t i = 0; i < m; i++)
    {
        std::cout << "Edge " << i << ": " << edges[i] << " " << weights[i] << std::endl;
    }

    dygrl::STGraph graph = dygrl::STGraph(n, m, offsets, edges, weights);

    

    // dygrl::CompressedWalks walks = dygrl::CompressedWalks(graph.offsets, graph.edges, graph.weights, 0);  
    // // Assign the head frequency we read
	// cout << endl << "Head frequency is " << compressed_lists::head_frequency << ", and thus, chunk size is " << compressed_lists::head_mask << endl;
    // // ---------------------------------

    // //初始化C树
    // dygrl::Wharf malin = dygrl::Wharf(n, m, offsets, edges);

    // auto produce_initial_walk_corpus = timer("WalkCorpusProduction", false); 
    // produce_initial_walk_corpus.start();
    // malin.generate_initial_random_walks();
    // auto walk_corpus_production_time =  produce_initial_walk_corpus.get_total();


    // // get walk path
    // std::cout << "Walk path test: " << std::endl;
    // for (size_t curWalkID = 0; curWalkID < n * w ; curWalkID++) {
    //     std::cout << malin.walk_simple_find(curWalkID) << std::endl;
    // }

    // // get vertedx test
    // std::cout << "Vertex test: " << std::endl;
    // for (size_t curVertexID = 0; curVertexID < n ; curVertexID++) {
    //     malin.get_neighbors(curVertexID);
    // }
}
int main(int argc, char** argv)
{
    std::cout << "Running experiment with: " << num_workers() << " threads." << std::endl;
    commandLine command_line(argc, argv, "");

    get_walkpath(command_line);
}


