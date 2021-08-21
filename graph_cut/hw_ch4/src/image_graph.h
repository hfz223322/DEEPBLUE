/**
______________________________________________________________________
*********************************************************************
* @brief  This file is developed for the course of ShenLan XueYuan:
* Fundamental implementations of Computer Vision
* all rights preserved
* @author Xin Jin, Zhaoran Wu
* @contact: xinjin1109@gmail.com, zhaoran.wu1@gmail.com
*
______________________________________________________________________
*********************************************************************
**/
#pragma once
#include "distribution.h"
#include "graph.h"
#include <opencv2/core.hpp>

/**
 * @brief get a id of a point
 *
 * @param row:  row of a point in an image
 * @param col:  col of a point in an image
 * @param step: cols of an image
 * @return int: id of a point (point (0,0) has id 0)
 */
inline int pos_to_id(int row, int col, int step);

/**
 * @brief get position(row col)of a point with specific id
 *
 * @param id : id of a point
 * @param step : cols of an image
 * @return std::pair<int, int> : position row and col
 */
inline std::pair<int, int> id_to_pos(int id, int step);
/**
 * @brief Edge of Image Graph design for Maximum-Flow computation
 *
 */
struct Edge : public EdgeBase {
    /**
     * @brief Construct a new Edge object
     *
     * @param weight : capcity of a edge
     */
    Edge(double weight);

    /**
     * @brief Get the residual : the remain space of a edge
     *
     * @return double :residual
     */
    double get_residual() const;
    /**
     * @brief if a edge has remain space
     */
    bool is_full();

    double cap_;   // capacity of a edge
    double flow_;  // flow on a edge
};

typedef NodeBase<Edge> Node;  // specification class for Node of ImageGraph

/**
 * @brief Class of ImageGraph with is build with src node, sink node and all
 * pixels
 *
 */
class ImageGraph : public Graph<Node, Edge> {
   public:
    /**
     * @brief Construct a new Image Graph object
     *
     * @param img : image to build the graph
     * @param points_foreground : scribbles of the foreground
     * @param points_background : scribbles of the background
     */
    ImageGraph(cv::Mat img, const std::vector<cv::Point>& points_foreground,
               const std::vector<cv::Point>& points_background);

    const int src_id_;   // id of the source node
    const int sink_id_;  // id of the sink node

   private:
    cv::Mat img_;        // original image in type CV_8UC3
    Distribution dist_;  // distribution of scribbles
};

/*--------------------------------------------------------
#####################implementation: inline function #####################
---------------------------------------------------------*/

inline int pos_to_id(int row, int col, int step) {
    return row * step + col;
}

inline std::pair<int, int> id_to_pos(int id, int step) {
    return {id / step, id % step};
}

inline double Edge::get_residual() const {
    return cap_ - flow_;
}

inline bool Edge::is_full() {
    return cap_ <= flow_;
}