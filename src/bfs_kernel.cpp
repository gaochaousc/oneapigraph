#include "dfs.h"

using namespace sycl;

void dfs_gpu(Vertex* vertex_arr, int vertex_arr_size, Edge* edge_arr, int edge_arr_size, bool* active_mask, bool* visited, int source, int* output)
{
  queue q{gpu_selector{}};
  const property_list props = property::buffer::use_host_ptr();
  buffer<Vertex, 1> d_vertex_arr(vertex_arr, vertex_arr_size, props);
  buffer<Edge, 1> d_edge_arr(edge_arr, edge_arr_size, props);
  buffer<bool, 1> d_active_mask(active_mask, vertex_arr_size, props);
  buffer<bool, 1> d_active_update(active_mask, vertex_arr_size, props);
  buffer<bool, 1> d_visited(visited, vertex_arr_size, props);
  buffer<bool, 1> d_done(true);
  buffer<int, 1> d_cost(0);
  buffer<int, 1> d_output(output, vertex_arr_size, props);
  
  bool done = true;
  
  while(!done)
  {
    // Reinitialize device done to True
    q.submit([&](handler & cgh){
      accessor done_acc(d_done, cgh);
      done_acc[0] = done;
    });
    
    q.submit([&](handler & cgh){
      accessor d_vertex_acc(d_vertex_arr, cgh);
      accessor d_edge_acc(d_edge_arr, cgh);
      accessor d_active_acc(d_active_mask, cgh);
      accessor d_active_update_acc(d_active_update, cgh);
      accessor d_visited_acc(d_visited, cgh);
      accessor d_output_acc(d_output, cgh);
      accessor d_cost_acc(d_cost, cgh);
      accessor done_acc(d_done, cgh);
      cgh.parallel_for(range<1>(edge_arr_size), [=](auto i){
        Edge edge = d_edge_acc[i];
        if (d_active_acc[edge.source] && !d_visited_acc[edge.dest])
        {
          d_visited_acc[edge.dest] = true;
          d_output_acc[edge.dest] = d_cost_acc[0]+1;
          done_acc[0] = false;
          d_active_update_acc[edge.dest] = true;
        }
      });

    });
    
    q.wait();
    
    q.submit([&](handler & cgh)
    {
      accessor d_active_acc(d_active_mask, cgh);
      accessor d_active_update_acc(d_active_update, cgh);
      cgh.parallel_for(range<1>(vertex_arr_size), [=](auto i){
        d_active_acc[i] = d_active_update_acc[i];
      });
    });
    
    
    
    // extract device done
    q.submit([&](handler & cgh){
      accessor done_acc(d_done, cgh);
      accessor d_cost_acc(d_cost, cgh);
      done = done_acc[0];
      d_cost_acc[0] += 1;
      
    });
  }
}