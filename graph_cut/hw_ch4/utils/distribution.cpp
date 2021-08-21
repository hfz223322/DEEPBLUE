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
#include "distribution.h"
#include "display.h"
#include <opencv2/core/core.hpp>
Distribution::Distribution(cv::Mat img, std::vector<cv::Point> foreground,
                           std::vector<cv::Point> background)
    : gmms_{GMM(img, foreground, 2), GMM(img, background, 2)},
      lamda_(5.0),
      img_(img.clone()) {
    for (int i = 0; i < gmms_.size(); i++) {
        gmms_[i].run(20);
    }
}

cv::Mat Distribution::get_probability_map(int id) {
    cv::Mat log_prob = cv::Mat::zeros(img_.size(), CV_64FC1);
    cv::log(gmms_[1 - id].get_prob(), log_prob);
    return -lamda_ * log_prob;
}
