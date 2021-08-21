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
#include "image_graph.h"
#include "distribution.h"
#include "opencv_utils.h"
#include <vector>

static const int dire[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

ImageGraph::ImageGraph(cv::Mat img,
                       const std::vector<cv::Point>& points_foreground,
                       const std::vector<cv::Point>& points_background)
    : Graph(img.rows * img.cols + 2),
      src_id_(0),
      sink_id_(img.rows * img.cols + 1),
      img_(img),
      dist_(img, points_foreground, points_background) {
    cv::Mat w_fore = dist_.get_probability_map(0);
    cv::Mat w_back = dist_.get_probability_map(1);
    // todo build a weighted graph for max flow algorithm.
    for(int i = 0; i < points_foreground.size(); i++)
    {
         w_fore.at<double>(points_foreground[i]) = 10000.0;
         w_back.at<double>(points_foreground[i]) = 0.0;
    } 
    for(int j = 0; j < points_background.size(); j++)
    {
         w_fore.at<double>(points_background[j]) = 0.0;
         w_back.at<double>(points_background[j]) = 10000.0;
    } 
    /***std::cout << "OK?" <<std::endl;
    double max_val;
    double min_val; 
    cv::minMaxLoc(w_back, &min_val, &max_val);
    while(1)
    {
         std::cout << max_val << std::endl << min_val << std::endl;
    }
    std::cout << "OK!" << std::endl;***/
    for(int i = 0; i < img.rows * img.cols; i++)
    {    
         std::pair<int, int> pos = id_to_pos(i, img_.cols);
         /***Edge node_edge_s(w_fore.at<double>(cv::Point(pos.second, pos.first)));***/
         add_binary_edge(src_id_, i+1, new Edge(w_fore.at<double>(cv::Point(pos.second, pos.first))));
         /***Edge node_edge_t(w_back.at<double>(cv::Point(pos.second, pos.first)));***/
         add_binary_edge(sink_id_, i+1, new Edge(w_back.at<double>(cv::Point(pos.second, pos.first))));    
    }
    for(int i = 0; i < img.rows * img.cols; i++)
    {    
         std::pair<int, int> pos = id_to_pos(i, img_.cols);
         cv::Vec3b s_rgb = img_.at<cv::Vec3b>(cv::Point(pos.second, pos.first));
         if((pos.first + 1) <= (img_.rows - 1))
         {
              cv::Vec3b d_rgb = img_.at<cv::Vec3b>(cv::Point(pos.second, pos.first + 1));  
              /***Edge node_edge(compute_weight(s_rgb, d_rgb, 2.5e-5));***/
              add_binary_edge(i+1, pos_to_id(pos.first + 1, pos.second, img_.cols) + 1, new Edge(compute_weight(s_rgb, d_rgb, 2.5e-5)));
              //std::cout << compute_weight(s_rgb, d_rgb, 2.5e-5) << std::endl;
         }
         if((pos.second + 1) <= (img_.cols - 1))
         {
              cv::Vec3b d_rgb = img_.at<cv::Vec3b>(cv::Point(pos.second + 1, pos.first));
              /***Edge node_edge(compute_weight(s_rgb, d_rgb, 2.5e-5));***/
              add_binary_edge(i+1, pos_to_id(pos.first, pos.second + 1, img_.cols) + 1, new Edge(compute_weight(s_rgb, d_rgb, 2.5e-5)));
              //std::cout << compute_weight(s_rgb, d_rgb, 2.5e-5) << std::endl;
         }          
    }
}

/*--------------------------------------------------------
#####################implementation: Edge #####################
---------------------------------------------------------*/
Edge::Edge(double weight) : EdgeBase(), cap_(weight), flow_(0.0) {
}
