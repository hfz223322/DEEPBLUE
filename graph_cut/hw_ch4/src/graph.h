/**
______________________________________________________________________
*********************************************************************
* @brief This file is developed for the course of ShenLan XueYuan:
* Fundamental implementations of Computer Vision
* all rights preserved
* @author Xin Jin, Zhaoran Wu
* @contact: xinjin1109@gmail.com, zhaoran.wu1@gmail.com
*
______________________________________________________________________
*********************************************************************
**/
#pragma once
#include <iostream>
#include <list>
#include <vector>
/**
 * @brief a template calss for a general node
 *
 * @tparam TypeEdge : the edge type of a node
 */
template <typename TypeEdge>
struct NodeBase {
    /**
     * @brief Construct a new Node Base object
     *
     * @param id : id of specific node
     */
    NodeBase(int id);

    /**
     * @brief add a node as the neighbour of current node with a specific edge
     *
     * @param target_node : neighbour node
     * @param edge : edge from current node to target node
     */
    void add_neighbour(NodeBase<TypeEdge>* target_node, TypeEdge* edge);
    /**
     * @brief back up a node with his parent (previous)
     *
     * @param prev : parent(previous) node
     * @param edge : edge from parent node to current node
     */
    void back_up_prev(NodeBase<TypeEdge>*, TypeEdge*);

    std::list<std::pair<NodeBase<TypeEdge>*, TypeEdge*>>
        neighbours_;  // all the neighbors of current node
    int id_;          // current node id
    std::pair<NodeBase<TypeEdge>*, TypeEdge*> prev_;  // info of previous node
};
/**
 * @brief base  class for a Edge
 *
 */
class EdgeBase {
   public:
    EdgeBase() = default;

   protected:
};
/**
 * @brief base class of a graph
 *
 * @tparam TypeNode
 * @tparam TypeEdge
 */
template <typename TypeNode, typename TypeEdge>
class Graph {
   public:
    /**
     * @brief Construct a new Graph object
     *
     * @param num_nodes num of node in a graph
     */
    Graph(int num_nodes);
    /**
     * @brief add a binary edge between 1 and 2, which means 1-->2 and 2-->1
     *
     * @param id_node1: id of first node
     * @param id_node2: id of second node
     * @param edge_weight : weight of the edge
     */
    void add_binary_edge(int id_node1, int id_node2, TypeEdge* edge_weight);
    /**
     * @brief add a unary edgy between src and target, which means src-->target
     *
     * @param id_src : id of src node
     * @param id_target :id of target ndoe
     * @param edge_weight : weight of the edge
     */
    void add_unary_edge(int id_src, int id_target, TypeEdge* edge_weight);
    /**
     * @brief Get the root of the whole graph
     *
     * @return TypeNode* point to the root node
     */
    TypeNode* get_root();

   protected:
    std::vector<TypeNode> nodes_;
};

/*--------------------------------------------------------
#####################implementation: Graph #####################
---------------------------------------------------------*/

template <typename TypeNode, typename TypeEdge>
Graph<TypeNode, TypeEdge>::Graph(int num_nodes) {
    for (int i = 0; i < num_nodes; i++) {
        nodes_.emplace_back(i);
    }
}

template <typename TypeNode, typename TypeEdge>
void Graph<TypeNode, TypeEdge>::add_binary_edge(int id_node1, int id_node2,
                                                TypeEdge* edge) {
    if (id_node1 < nodes_.size() && id_node2 < nodes_.size()) {
        nodes_[id_node1].add_neighbour(&nodes_[id_node2], edge);
        nodes_[id_node2].add_neighbour(&nodes_[id_node1], edge);
    } else {
        std::cout << "node id is over maximum ! addition will be ignored."
                  << std::endl;
    }
}

template <typename TypeNode, typename TypeEdge>
void Graph<TypeNode, TypeEdge>::add_unary_edge(int id_src, int id_target,
                                               TypeEdge* edge) {
    nodes_[id_src].add_neighbour(&nodes_[id_target], edge);
}

template <typename TypeNode, typename TypeEdge>
TypeNode* Graph<TypeNode, TypeEdge>::get_root() {
    return &nodes_[0];
}

/*--------------------------------------------------------
#####################implementation: NodeBase #####################
---------------------------------------------------------*/
template <typename TypeEdge>
NodeBase<TypeEdge>::NodeBase(int id) : id_(id), prev_{nullptr, nullptr} {
}

template <typename TypeEdge>
inline void NodeBase<TypeEdge>::add_neighbour(NodeBase<TypeEdge>* target_node,
                                              TypeEdge* edge) {
    neighbours_.emplace_back(target_node, edge);
}

template <typename TypeEdge>
inline void NodeBase<TypeEdge>::back_up_prev(NodeBase<TypeEdge>* prev_node,
                                             TypeEdge* edge) {
    prev_.first = prev_node;
    prev_.second = edge;
}