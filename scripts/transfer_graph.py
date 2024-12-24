import sys
import random

def read_graph(file_path):
    total_edges = 0
    total_verts = 0
    adjacency_list = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 跳过注释行
            if line.startswith('#'):
                continue
            
            from_node, to_node = map(int, line.strip().split())
            total_verts = max(total_verts, from_node, to_node)
            if from_node not in adjacency_list:
                adjacency_list[from_node] = []
            adjacency_list[from_node].append(to_node)
            total_edges += 1
    
    return adjacency_list, total_verts + 1, total_edges

def transfer_graph(file_path, output_file_path, max_weight = 100):
    graph, total_verts, total_edges = read_graph(file_path)
    #写入文件
    with open(output_file_path, 'w') as output_file:
        output_file.write(str(total_verts) + "\n")
        output_file.write(str(total_edges) + "\n")
        for src in graph:
            for dst in graph[src]:
                # output_file.write(f"{src} {dst} {random.randint(a=1, b=max_weight)}\n")
                output_file.write(f"{src} {dst} {random.randint(a=1, b=max_weight) + total_verts * dst}\n")
        
if __name__ == "__main__":    
    file_path = sys.argv[1]              # 输入文件路径
    output_file_path = sys.argv[2]       # 输出文件路径
    transfer_graph(file_path, output_file_path)