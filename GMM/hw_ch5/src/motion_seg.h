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

#include "gmm.h"
#include <opencv2/core/core.hpp>

/**
 * @brief Motion segementation using gmm
 *
 */
class MotionSeg {
   public:
    /**
     * @brief Construct a new Motion Seg object
     *
     * @param rows :rows of the image
     * @param cols :cols of the image
     * @param num_gaussian :num of the gaussian model
     * @param config: parameter of gmm
     */
    MotionSeg(int rows, int cols, int num_gaussian, const gmm::ConfigParam& config);
    /**
     * @brief process a image sequence with motion segementation
     *
     * @param video :image sequence
     */
    void process(const std::vector<cv::Mat>& video);

   private:
    std::vector<gmm::GMM> gmm_map;  // vector of gmm(each pixel corresponding with one element)
};