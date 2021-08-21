#include "gvf.h"
#include "display.h"
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
/**
 * @brief Construct a new Param G V F:: Param G V F object
 *
 * @param smooth_term_weight: weight of smooth term (mu)
 * @param sigma:
 * @param init_step_size
 */
ParamGVF::ParamGVF(double smooth_term_weight, double init_step_size)
    : smooth_term_weight_(smooth_term_weight), init_step_size_(init_step_size) {
}

/**
 * @brief Construct a new GVF::GVF object
 *
 * @param grad_original_x : the gradient of original image in the x
 * direction
 * @param grad_original_y : the gradient of original image in the y
 * direction
 * @param param_gvf: the prameter set of gvf
 */
GVF::GVF(cv::Mat grad_original_x, cv::Mat grad_original_y,
         const ParamGVF& param_gvf)
    : GradientDescentBase(param_gvf.init_step_size_),
      param_gvf_(param_gvf),
      data_term_weight_(cv::Mat::zeros(grad_original_x.size(), CV_64F)),
      laplacian_gvf_x_(cv::Mat::zeros(grad_original_x.size(), CV_64F)),
      laplacian_gvf_y_(cv::Mat::zeros(grad_original_y.size(), CV_64F)) {
    // initialize the gvf external energy
    cv::Mat square_grad_original_x, square_grad_original_y;
    cv::pow(grad_original_x, 2.0f, square_grad_original_x);
    cv::pow(grad_original_y, 2.0f, square_grad_original_y);

    cv::Mat mag_original;
    cv::sqrt(square_grad_original_x + square_grad_original_y, mag_original);
    cv::GaussianBlur(mag_original, mag_original, cv::Size(3, 3), 3, 3);
    cv::Sobel(mag_original, gvf_initial_x_, CV_64F, 1, 0, 3);
    cv::Sobel(mag_original, gvf_initial_y_, CV_64F, 0, 1, 3);

    // compute the date term weight
    cv::Mat square_gvf_initial_x, square_gvf_initial_y;
    cv::pow(gvf_initial_x_, 2.0f, square_gvf_initial_x);
    cv::pow(gvf_initial_y_, 2.0f, square_gvf_initial_y);
    data_term_weight_ = square_gvf_initial_x + square_gvf_initial_y;
}
/**
 * @brief initialize the gvf: Hits: there are different ways for initialization
 *        1. use external energy, such as gradient of image, or add some other
 * term, namely line, edge, curvature
 *        2. use grad||grad(img)|| to make the vector field towards to the edge
 */
void GVF::initialize() {
    gvf_x_ = gvf_initial_x_.clone();
    gvf_y_ = gvf_initial_y_.clone();
}

/**
 * @brief update the gvf accodging to Euler-lagrange Equation
 *
 */
void GVF::update() {
    // TODO: update the gvf after in iteration (ppt page : 24)
    cv::Laplacian(gvf_x_, laplacian_gvf_x_, CV_64F);
    cv::Laplacian(gvf_y_, laplacian_gvf_y_, CV_64F);
    gvf_x_ += step_size_ * ((param_gvf_.smooth_term_weight_ * laplacian_gvf_x_) - (gvf_x_ - gvf_initial_x_).mul(data_term_weight_));
    gvf_y_ += step_size_ * ((param_gvf_.smooth_term_weight_ * laplacian_gvf_y_) - (gvf_y_ - gvf_initial_y_).mul(data_term_weight_));
    display_gvf(gvf_x_, gvf_y_, 1, false);
}
/**
 * @brief compute enegy according to current gvf
 *
 * @return double
 */
double GVF::compute_energy() {
    // TODO : compute current energy (ppt page : 23)
    // compute data term energy
    float smooth_term_energy;
    // compute smooth term energy
    double data_term_energy;
    cv::Mat gvf_x_dx, gvf_x_dy, gvf_y_dx, gvf_y_dy;
    cv::Sobel(gvf_x_, gvf_x_dx, CV_64F, 1, 0);
    cv::Sobel(gvf_x_, gvf_x_dy, CV_64F, 0, 1);
    cv::Sobel(gvf_y_, gvf_y_dx, CV_64F, 1, 0);
    cv::Sobel(gvf_y_, gvf_y_dy, CV_64F, 0, 1);
    cv::pow(gvf_x_dx, 2.0f, gvf_x_dx);
    cv::pow(gvf_x_dy, 2.0f, gvf_x_dy);
    cv::pow(gvf_y_dx, 2.0f, gvf_y_dx);
    cv::pow(gvf_y_dy, 2.0f, gvf_y_dy);
    smooth_term_energy = cv::sum(param_gvf_.smooth_term_weight_ * (gvf_x_dx + gvf_x_dy + gvf_y_dx + gvf_y_dy))[0]; 
    cv::Mat gvf_data_energy_x, gvf_data_energy_y;
    cv::pow(gvf_x_ - gvf_initial_x_, 2.0f, gvf_data_energy_x);
    cv::pow(gvf_y_ - gvf_initial_y_, 2.0f, gvf_data_energy_y);
    data_term_energy = cv::sum((gvf_data_energy_x + gvf_data_energy_y).mul(data_term_weight_))[0];
    return smooth_term_energy + data_term_energy;
}

/**
 * @brief roll back gvf result
 *
 */
void GVF::roll_back_state() {
    gvf_x_ = last_gvf_x_;
    gvf_y_ = last_gvf_y_;
}
/**
 * @brief back up gvf result in between
 *
 */
void GVF::back_up_state() {
    last_gvf_x_ = gvf_x_.clone();
    last_gvf_y_ = gvf_y_.clone();
}
/**
 * @brief get gvf result: gvf_x_ and gvf_y_
 *
 * @return std::vector<cv::Mat> save them in a vector
 */
std::vector<cv::Mat> GVF::get_result_gvf() const {
    std::vector<cv::Mat> gvf_result(2);
    gvf_result[0] = gvf_x_.clone();
    gvf_result[1] = gvf_y_.clone();
    return gvf_result;
}
/**
 * @brief print when terminate
 *
 */
void GVF::print_terminate_info() const {
    std::cout << "GVF iteration finished." << std::endl;
}

std::string GVF::return_drive_class_name() const {
    return "GVF";
}