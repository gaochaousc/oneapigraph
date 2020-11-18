#include "bfs.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <limits.h>

#define MAX_THREADS_PER_BLOCK 256

using namespace sycl;

#define NUM_OF_TEST 100

#define USE_GPU 1
#define USE_FPGA 0

bool Initialize(std::string edge_file, std::string vertex_file, \
                      int & no_of_node, Node* & node_arr, int & no_of_edge, Edge* & edge_arr, \
                      bool* &active_mask, bool* &updating_active_mask, bool* &visited, int* &cost, int* &cost_device)
{
  /* Input Format */
  /*
    Edge File:
    num_of_edge
    src dest
    src dest
    ......

    Vertex File:
    num_of_vertex
    starting_edge no_of_edge 
    ...
    Note that starting_edge = -1 for vertex without any outgoing edge
  */

  std::ifstream edge, vertex;
  edge.open(edge_file);
  vertex.open(vertex_file);
  
  if (!edge.is_open() || !vertex.is_open())
  {
    std::cout << "vertex file not found" << std::endl;
    return false;
  }

  char buffer[128];
  edge >> buffer;
  no_of_edge = std::stoi(buffer);
  vertex >> buffer;
  no_of_node = std::stoi(buffer);

  edge_arr = (Edge*)malloc(no_of_edge*sizeof(Edge));
  node_arr = (Node*)malloc(no_of_node*sizeof(Node));
  active_mask = (bool*)malloc(no_of_node*sizeof(bool));
  updating_active_mask = (bool*)malloc(no_of_node*sizeof(bool));
  visited = (bool*)malloc(no_of_node*sizeof(bool));
  cost = (int*)malloc(no_of_node*sizeof(int));
  cost_device = (int*)malloc(no_of_node * sizeof(int));


  for (int i = 0; i < no_of_edge; i++)
  {
    edge >> buffer;
    edge_arr[i].source = std::stoi(buffer);
    edge >> buffer;
    edge_arr[i].dest = std::stoi(buffer);
    edge_arr->weight = 1;
  }
  for (int i = 0; i < no_of_node; i++)
  {
    vertex >> buffer;
    node_arr[i].first_edge = std::stoi(buffer);
    vertex >> buffer;
    node_arr[i].no_of_edges = std::stoi(buffer);
    active_mask[i] = false;
    updating_active_mask[i] = false;
    visited[i] = false;
    cost[i] = INT_MAX;
    cost_device[i] = INT_MAX;
  }
  
  edge.close();
  vertex.close();
  return true;
}

//This code is adapted from https://github.com/zjin-lcf/oneAPI-DirectProgramming and then modified

void bfs_cpu(int no_of_nodes, Node *graph_nodes, int edge_list_size, \
    Edge *graph_edges, bool *graph_mask, bool *updating_graph_mask, \
    bool *graph_visited, int *cost_ref, int source){
  
  graph_mask[source] = true;
  graph_visited[source] = true;
  cost_ref[source] = 0;
  char stop;
  int k = 0;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (graph_nodes[tid].first_edge == -1)
        continue;

      if (graph_mask[tid]){ 
        graph_mask[tid]=false;
        for(int i=graph_nodes[tid].first_edge; i<(graph_nodes[tid].no_of_edges + graph_nodes[tid].first_edge); i++){
          int id = graph_edges[i].dest;  //--cambine: node id is connected with node tid
          if(!graph_visited[id]){  //--cambine: if node id has not been visited, enter the body below
            cost_ref[id]=cost_ref[tid]+1;
            updating_graph_mask[id] = true;;
          }
        }
      }    
    }

    for(int tid=0; tid< no_of_nodes ; tid++ )
    {
      if (updating_graph_mask[tid]){
        graph_mask[tid]=true;
        graph_visited[tid]=true;
        stop=1;
        updating_graph_mask[tid]=false;
      }
    }
    k++;
  }
  while(stop);
}

void bfs_device(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
    Edge *h_graph_edges, bool *h_graph_mask, bool *h_updating_graph_mask, \
    bool *h_graph_visited, int *h_cost, int source){

    char h_over;
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;
    h_cost[source] = 0;
    
  {
  #if USE_GPU
    queue q(gpu_selector {});
  #elif USE_FPGA
    queue q(INTEL::fpga_emulator_selector {});
  #else
    queue q(cpu_selector {});
  #endif
    const property_list props = property::buffer::use_host_ptr();
    buffer<Node,1> d_graph_nodes(h_graph_nodes, no_of_nodes, props);
    buffer<Edge,1> d_graph_edges(h_graph_edges, edge_list_size, props);
    buffer<bool,1> d_graph_mask(h_graph_mask, no_of_nodes, props);
    buffer<bool,1> d_updating_graph_mask(h_updating_graph_mask, no_of_nodes, props);
    buffer<bool,1> d_graph_visited(h_graph_visited, no_of_nodes, props);
    buffer<int,1> d_cost(h_cost, no_of_nodes, props);

    buffer<char,1> d_over(1);

    int global_work_size = (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK;
    
    //--2 invoke kernel
    do {
      h_over = 0;
      q.submit([&](handler& cgh) {
          auto d_over_acc = d_over.get_access<access::mode::write>(cgh);
          cgh.copy(&h_over, d_over_acc);
      });
      q.submit([&](handler& cgh) {
          auto d_graph_nodes_acc = d_graph_nodes.get_access<access::mode::read>(cgh);
          auto d_graph_edges_acc = d_graph_edges.get_access<access::mode::read>(cgh);
          auto d_graph_mask_acc = d_graph_mask.get_access<access::mode::write>(cgh);
          auto d_updating_graph_mask_acc = d_updating_graph_mask.get_access<access::mode::write>(cgh);
          auto d_graph_visited_acc = d_graph_visited.get_access<access::mode::read>(cgh);
          auto d_cost_acc = d_cost.get_access<access::mode::write>(cgh);

          cgh.parallel_for<class kernel1>( nd_range<1>(range<1>(global_work_size),
                range<1>(MAX_THREADS_PER_BLOCK)), [=] (nd_item<1> item) {
              int tid = item.get_global_id(0);
              if (d_graph_nodes_acc[tid].first_edge != -1)
              {
                if( tid<no_of_nodes && d_graph_mask_acc[tid]){
                  d_graph_mask_acc[tid]=false;
                  for(int i=d_graph_nodes_acc[tid].first_edge; 
                      i<(d_graph_nodes_acc[tid].no_of_edges + d_graph_nodes_acc[tid].first_edge); i++){
                    int id = d_graph_edges_acc[i].dest;
                    if(!d_graph_visited_acc[id]){
                      d_cost_acc[id]=d_cost_acc[tid]+1;
                      d_updating_graph_mask_acc[id]=true;
                    }
                  }
                } 
              }
                 
          });
      });

      q.wait();

      //--kernel 1

      q.submit([&](handler& cgh) {

          auto d_graph_mask_acc = d_graph_mask.get_access<access::mode::write>(cgh);
          auto d_updating_graph_mask_acc = d_updating_graph_mask.get_access<access::mode::read_write>(cgh);
          auto d_graph_visited_acc = d_graph_visited.get_access<access::mode::write>(cgh);
          auto d_over_acc = d_over.get_access<access::mode::write>(cgh);

          cgh.parallel_for<class kernel2>( nd_range<1>(range<1>(global_work_size),
                range<1>(MAX_THREADS_PER_BLOCK)), [=] (nd_item<1> item) {
              int tid = item.get_global_id(0);
              if( tid<no_of_nodes && d_updating_graph_mask_acc[tid]){

              d_graph_mask_acc[tid]=true;
              d_graph_visited_acc[tid]=true;
              d_over_acc[0]=1;
              d_updating_graph_mask_acc[tid]=false;
              }
          });
      });

      q.submit([&](handler& cgh) {
          auto d_over_acc = d_over.get_access<access::mode::read>(cgh);
          cgh.copy(d_over_acc, &h_over);
      });
      q.wait();
    }
    while (h_over);

  }
}

// The edge centric version of device bfs
// Apparently the vertex centric bfs has explored enough parallelism for the device used,
// So the ec version fails to obtain speedup, deprecated

void bfs_device_ec(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
    Edge *h_graph_edges, bool *h_graph_mask, bool *h_updating_graph_mask, \
    bool *h_graph_visited, int *h_cost, int source){

    char h_over;
    h_graph_mask[source] = true;
    h_graph_visited[source] = true;
    h_cost[source] = 0;
    
  {
  #if USE_GPU
    queue q(gpu_selector {});
  #elif USE_FPGA
    queue q(INTEL::fpga_emulator_selector {});
  #else
    queue q(cpu_selector {});
  #endif
    const property_list props = property::buffer::use_host_ptr();
    buffer<Node,1> d_graph_nodes(h_graph_nodes, no_of_nodes, props);
    buffer<Edge,1> d_graph_edges(h_graph_edges, edge_list_size, props);
    buffer<bool,1> d_graph_mask(h_graph_mask, no_of_nodes, props);
    buffer<bool,1> d_updating_graph_mask(h_updating_graph_mask, no_of_nodes, props);
    buffer<bool,1> d_graph_visited(h_graph_visited, no_of_nodes, props);
    buffer<int,1> d_cost(h_cost, no_of_nodes, props);

    buffer<char,1> d_over(1);

    int global_work_size_node = (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK;
    int global_work_size_edge = (edge_list_size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK;

    //--2 invoke kernel
    do {
      h_over = 0;
      q.submit([&](handler& cgh) {
          auto d_over_acc = d_over.get_access<access::mode::write>(cgh);
          cgh.copy(&h_over, d_over_acc);
      });
      q.submit([&](handler& cgh) {
          auto d_graph_nodes_acc = d_graph_nodes.get_access<access::mode::read>(cgh);
          auto d_graph_edges_acc = d_graph_edges.get_access<access::mode::read>(cgh);
          auto d_graph_mask_acc = d_graph_mask.get_access<access::mode::write>(cgh);
          auto d_updating_graph_mask_acc = d_updating_graph_mask.get_access<access::mode::write>(cgh);
          auto d_graph_visited_acc = d_graph_visited.get_access<access::mode::read>(cgh);
          auto d_cost_acc = d_cost.get_access<access::mode::write>(cgh);

          cgh.parallel_for<class kernel1>( nd_range<1>(range<1>(global_work_size_edge),
                range<1>(MAX_THREADS_PER_BLOCK)), [=] (nd_item<1> item) {
              int tid = item.get_global_id(0);
              if (tid<edge_list_size)
              {
                int src = d_graph_edges_acc[tid].source;
                int dst = d_graph_edges_acc[tid].dest;
                if(d_graph_mask_acc[src]){
                  //d_graph_mask_acc[src]=false;
                  if(!d_graph_visited_acc[dst]){
                    d_cost_acc[dst]=d_cost_acc[src]+1;
                    d_updating_graph_mask_acc[dst]=true;
                  }
                } 
              }
          });
      });
      
      q.wait();

      //--kernel 1

      q.submit([&](handler& cgh) {

          auto d_graph_mask_acc = d_graph_mask.get_access<access::mode::write>(cgh);
          auto d_updating_graph_mask_acc = d_updating_graph_mask.get_access<access::mode::read_write>(cgh);
          auto d_graph_visited_acc = d_graph_visited.get_access<access::mode::write>(cgh);
          auto d_over_acc = d_over.get_access<access::mode::write>(cgh);

          cgh.parallel_for<class kernel2>( nd_range<1>(range<1>(global_work_size_node),
                range<1>(MAX_THREADS_PER_BLOCK)), [=] (nd_item<1> item) {
              int tid = item.get_global_id(0);
              if( tid<no_of_nodes ){
                if (d_updating_graph_mask_acc[tid])
                {
                  d_graph_mask_acc[tid]=true;
                  d_graph_visited_acc[tid]=true;
                  d_over_acc[0]=1;
                  d_updating_graph_mask_acc[tid]=false;
                }
                else if (d_graph_mask_acc[tid])
                {
                  d_graph_mask_acc[tid] = false;
                }
              }
          });
      });

      q.submit([&](handler& cgh) {
          auto d_over_acc = d_over.get_access<access::mode::read>(cgh);
          cgh.copy(d_over_acc, &h_over);
      });
      q.wait();
    }
    while (h_over);

  }
}

// Utility functions
bool Compare(int* cost_cpu, int* cost_device, int no_of_node)
{
  for(int i = 0; i < no_of_node; i++)
  {
    if (cost_cpu[i] != cost_device[i])
        return false;
  }
  return true;
}

void DisplayTestInfo(int no_of_node, int no_of_edge)
{
  #if USE_GPU
    queue q(gpu_selector {});
  #elif USE_FPGA
    queue q(INTEL::fpga_emulator_selector {});
  #else
    queue q(cpu_selector {});
  #endif
  std::cout << "Using Device: " << q.get_device().get_info<info::device::name>() << std::endl;
  std::cout << "Graph Dimension: " <<  no_of_node << " nodes " << no_of_edge << " edges " << std::endl;
  std::cout << "Running " << NUM_OF_TEST << " BFS test" << std::endl;
}


int main()
{
  int no_of_edge, no_of_node;
  Edge* edge_arr;
  Node* node_arr;
  bool* active_mask, * updating_active_mask, * visited;
  int * cost, *cost_device;
  int source;
  double cpu_time, gpu_time;

  if (!Initialize("edge.txt", "node.txt", no_of_node, node_arr, no_of_edge, edge_arr, \
                active_mask, updating_active_mask, visited, cost, cost_device))
  {
    std::cout << "Initialization failed" << std::endl;
    return 0;
  }
  
  //warm up
  source = 0;
  bfs_device(no_of_node, node_arr, no_of_edge, edge_arr, active_mask, updating_active_mask, visited, cost_device, source);
  
  DisplayTestInfo(no_of_node, no_of_edge);
  
  for (int k = 0; k < NUM_OF_TEST; k++)
  {
    // Choose a random source
    source = rand() % no_of_node;
    
    //Reset Result
    for (int i = 0; i < no_of_node; i++)
    {
      active_mask[i] = false;
      updating_active_mask[i] = false;
      visited[i] = false;
      cost[i] = INT_MAX;
    }
    /****** CPU BFS Computation******/
    dpc_common::TimeInterval t_serial;
    bfs_cpu(no_of_node, node_arr, no_of_edge, edge_arr, active_mask, updating_active_mask, visited, cost, source);
    cpu_time += t_serial.Elapsed();
    
    // Reset results
    for (int i = 0; i < no_of_node; i++)
    {
      active_mask[i] = false;
      updating_active_mask[i] = false;
      visited[i] = false;
      cost_device[i] = INT_MAX;
    }
    
    /****** Device BFS Computation******/
    dpc_common::TimeInterval t_parallel;
    bfs_device(no_of_node, node_arr, no_of_edge, edge_arr, active_mask, updating_active_mask, visited, cost_device, source);
    //bfs_device_ec(no_of_node, node_arr, no_of_edge, edge_arr, active_mask, updating_active_mask, visited, cost_device, source);
    gpu_time += t_parallel.Elapsed();
    
    if (!Compare(cost, cost_device, no_of_node))
    {
      std::cout << "Result diffs with Source=" << source << std::endl;
    }
  }
  
  std::cout << "Average Serial BFS Time: " << cpu_time / NUM_OF_TEST << std::endl;
  std::cout << "Average Parallel BFS Time: " << gpu_time / NUM_OF_TEST << std::endl;

  return 0;
}

