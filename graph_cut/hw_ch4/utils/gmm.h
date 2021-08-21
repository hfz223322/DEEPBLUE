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
// GMM template
// template <typename TFeature, typename DFeature, typename NGauss>
// In this case, fit a 3D GMM with 2 Gaussian
#pragma once
#include "em.h"
#include <opencv2/core/core.hpp>
#include <tuple>
#include <vector>

/**
 * @brief Gaussian3D class which offer some methods to build a single gaussian
 * distribution
 *
 */
class Gaussian3D {
   public:
    /**
     * @brief Construct a new Gaussian3D object using default constructor
     *
     */
    Gaussian3D() = default;
    /**
     * @brief Construct a new Gaussian3D object given expectation and covariance
     *
     * @param miu
     * @param sigma
     */
    Gaussian3D(const cv::Matx31d& miu, const cv::Matx33d& sigma);
    /**
     * @brief compute pdf given an image
     *
     * @param img
     * @return cv::Mat
     */
    cv::Mat compute_gaussian_pdf_map(cv::Mat img);
    /**
     * @brief Get expectation for a gaussian model
     *
     * @return cv::Matx31d
     */
    cv::Matx31d get_miu() const;
    /**
     * @brief Get covariance for a gaussian model
     *
     * @return cv::Matx33d
     */
    cv::Matx33d get_sigma() const;
    /**
     * @brief Set expectation for a gaussian model
     *
     * @param miu
     */
    void set_miu(const cv::Matx31d& miu);
    /**
     * @brief Set covariance for a gaussian model
     *
     * @param sigma
     */
    void set_sigma(const cv::Matx33d& sigma);
    /**
     * @brief Compute pdf given scribbles in the interaction tool
     *
     * @param samples
     * @return cv::Mat
     */
    cv::Mat compute_dataset_gaussian_pdf_map(cv::Mat samples);

   private:
    /**
     * @brief Compute single gaussian pdf given data
     *
     * @param data
     * @return double
     */
    double compute_gaussian_pdf(const cv::Matx31d& data);

    cv::Matx31d miu_;
    cv::Matx33d sigma_;
};

/**
 * @brief enum class
 *
 */
enum class GMM_MODE { IMAGE_MODE = 1, SCRIBBLE_MODE = 2 };

/**
 * @brief GMM class which is based on EM class
 *
 */
class GMM : public EMBase {
   public:
    /**
     * @brief Construct a new GMM object given image and number of gaussian
     * model
     *
     * @param img
     * @param num_gaussian_model
     */
    GMM(cv::Mat img, int num_gaussian_model);
    /**
     * @brief Construct a new GMM object given image, scribble and number of
     * gaussian model
     *
     * @param img
     * @param scribble
     * @param num_gaussian_model
     */
    GMM(cv::Mat img, const std::vector<cv::Point>& scribble,
        int num_gaussian_model);

    /**
     * @brief Get posterior, evaluate the p, intuitively this is the prob of the
     * samples being assigned to each of the k clusters
     *
     * @param id_model
     * @return cv::Mat
     */
    cv::Mat get_posterior(int id_model);
    /**
     * @brief Get GMM probability of given image w.r.t the updated param
     *
     * @return cv::Mat
     */
    cv::Mat get_prob();

   private:
    void initialize() override;
    void update_e_step() override;
    void update_m_step() override;

    /**
     * @brief update mean value of M step
     *
     * @param id_model
     * @param Nk
     */
    void update_miu(int id_model, double Nk);
    /**
     * @brief update covariance of M step
     *
     * @param id_model
     * @param Nk
     */
    void update_sigma(int id_model, double Nk);
    /**
     * @brief Update weight of M step
     *
     * @param id_model
     * @param Nk
     */
    void update_weight(int id_model, double Nk);

    cv::Mat img_;      // img is original image, whose size is m*n
    cv::Mat samples_;  // samples is a long vector, whose size is mn*1

    std::vector<double> w_gaussian_model_;      // weights of gaussian model
    std::vector<Gaussian3D> gaussian3d_model_;  // vector of gaussian model
    std::vector<cv::Mat> posterior_;

    GMM_MODE mode_;
};