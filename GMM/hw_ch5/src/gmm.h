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
#include <queue>
#include <vector>

namespace gmm {
/**
 * @brief Hyperparameter of motion based GMM model
 * @ a is for checking if the sample is in the k-th model ||sample-mean||<a*var
 * @ alpha is learning rate weight = (1-alpha) * weight + alpha * Mkt Mkt is an indicator, tells us which model the
 *   sample belongs to T is
 * @ T is foreground threshold, (sum from 1 to B of weight) > T, we use number B to check if the sample is in the fore-
 *   or background
 */
struct ConfigParam {
    ConfigParam(double a, double alpha, double T);
    double a_;
    double alpha_;
    double T_;
};
/**
 * @brief Gaussian paramater including the corresponding weight
 *
 */
struct GaussianParam {
    /**
     * @brief Construct a new Gaussian Param object by default
     *
     */
    GaussianParam() = default;
    /**
     * @brief Construct a new Gaussian Param object by parameters
     *
     * @param mean
     * @param var
     * @param weight
     */
    GaussianParam(double mean, double var, double weight);
    double mean_ = -100.0;
    double var_ = 50;
    double weight_ = 0.2;
};

/**
 * @brief overloding "<" for priority
 *
 * @param lhs
 * @param rhs
 * @return true
 * @return false
 */
bool operator<(const gmm::GaussianParam& lhs, const gmm::GaussianParam& rhs);

/**
 * @brief GMM model parameter including multiple gaussian parameters and some related functions
 *
 */
struct ModelParam {
    /**
     * @brief Construct a new Model Param object by number of gaussian
     *
     * @param num_gaussian
     */
    ModelParam(int num_gaussian);

    std::vector<GaussianParam> param_;

    /**
     * @brief sort with priority by weight/var, bigger at the end
     *
     */
    void sort_with_priority();

    /**
     * @brief sort with weight from big to small
     *
     */
    void sort_with_weight();
    /**
     * @brief normalize the weight so that the sum of it is 1
     *
     */
    void normalize_weight();
};
/**
 * @brief overloding << for print model parameters
 *
 * @param os
 * @param model_param
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os, const gmm::ModelParam& model_param);

/**
 * @brief class of Gaussian mixture model
 *
 */
class GMM {
   public:
    /**
     * @brief Construct a new GMM object by number of gaussian and config parameters
     *
     * @param num_gaussian
     * @param config_param
     */
    GMM();

    /**
     * @brief add sample into the class GMM
     *
     * @param sample
     */

    void add_sample(double sample);

    /**
     * @brief model parameter of GMM
     *
     * @return ModelParam&
     */
    ModelParam& model_param();

    /**
     * @brief tell if the sample is in fore- or background
     *
     * @param sample
     * @return true
     * @return false
     */
    bool is_in_foreground(double sample);
    static void set_config(const ConfigParam& config);
    static void set_num_gaussian(int num);

   private:
    /**
     * @brief Tell if the sample is in the id_model-th model
     *
     * @param sample
     * @param id_model
     * @return true
     * @return false
     */
    bool is_in_model(double sample, int id_model) const;
    /**
     * @brief Get the GM id object of sample
     *
     * @param sample
     * @return int
     */
    int get_gm_id(double sample);
    /**
     * @brief Remove the gaussian model with lowest priority and add the sample in gaussian model
     *
     * @param sample
     */
    void replace_model(double sample);
    /**
     * @brief update GMM, mean, var and weight
     *
     * @param sample
     * @param id
     */
    void update_gmm(double sample, int id);

    ModelParam model_param_;
    static int num_gaussians_;
    static ConfigParam config_param_;
};

}  // namespace gmm
