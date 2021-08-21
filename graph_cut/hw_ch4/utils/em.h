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
/**
 * @brief Expectation Maximazation Base class
 *
 */
class EMBase {
   public:
    EMBase() = default;
    /**
     * @brief Execute function for EM Algorithm
     *
     * @param max_iteration
     */
    void run(int max_iteration);

   protected:
    /**
     * @brief initialize the mean, covariances and weights
     *
     */
    virtual void initialize() = 0;
    /**
     * @brief Evaluate the posterior p, intuitively this is the prob of given
     * samples being assigned to each of the k clusters
     *
     */
    virtual void update_e_step() = 0;
    /**
     * @brief Estimate the parameters using MLE
     *
     */
    virtual void update_m_step() = 0;

    /**
     * @brief print terminate information
     *
     */
    virtual void print_terminate_info() const;
};