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
#include "gmm.h"
#include <array>
#include <opencv2/core/core.hpp>
#include <vector>
/**
 * @brief Distribution class, given an image, fore and background scribble to
 * get corresponding pdfs
 *
 */
class Distribution {
   public:
    /**
     * @brief Construct a new Distribution object
     *
     * @param img
     * @param foreground
     * @param background
     */
    Distribution(cv::Mat img, std::vector<cv::Point> foreground,
                 std::vector<cv::Point> background);
    /**
     * @brief Get the probability map of fore or background,  0 is foreground, 1
     * is background
     *
     * @param id
     * @return cv::Mat
     */
    cv::Mat get_probability_map(int id);

   private:
    std::array<GMM, 2> gmms_;
    double lamda_;
    cv::Mat img_;
};

/*--------------------------------------------------------
#####################implementation: inline #####################
---------------------------------------------------------*/
/**
 * @brief compute similarity of two pixels
 *
 * @param color1
 * @param color2
 * @param sigma_square_inv
 * @return double
 */
inline double compute_weight(const cv::Vec3f& color1, const cv::Vec3f& color2,
                             double sigma_square_inv = 2.5e-5) {
    cv::Vec3f diff = color2 - color1;
    return 20 * exp(-0.5 * sigma_square_inv * (diff.dot(diff)));
}