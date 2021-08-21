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
#include "motion_seg.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

int main(int argc, char** argv) {
    std::vector<cv::Mat> video;
   // for (int id = 1; id < 220; id++) {
       // cv::Mat img = cv::imread(argv[1] + std::to_string(id) + ".bmp", cv::IMREAD_GRAYSCALE);
       // video.push_back(img);
   // }
    for (int id = 1; id < 104; id++) {
        cv::Mat img = cv::imread(argv[1] + std::to_string(id) + ".jpg", cv::IMREAD_GRAYSCALE);
        video.push_back(img);
     }

    double a = 2.5;
    double alpha = 0.005;
    double T = 0.5;

    gmm::ConfigParam config_param(a, alpha, T);

    int num_gausian = 4;
    MotionSeg ms(video[0].rows, video[0].cols, num_gausian, config_param);
    ms.process(video);

    return 0;
}
