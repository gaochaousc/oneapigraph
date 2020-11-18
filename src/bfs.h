
#ifndef _DFS_H_
#define _DFS_H_

#include "dpc_common.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

struct Node
{
  int first_edge;
  int no_of_edges;
  
  Node(int first, int number) : first_edge(first), no_of_edges(number){}
};

struct Edge
{
  int source;
  int dest;
  int weight;
  
  Edge(int source_id, int dest_id, int w) : source(source_id), dest(dest_id), weight(w){}
};

void bfs_cpu(int no_of_nodes, Node *graph_nodes, int edge_list_size, Edge *graph_edges, char *graph_mask, char *updating_graph_mask, char *graph_visited, int *cost_ref, int source);

#endif
