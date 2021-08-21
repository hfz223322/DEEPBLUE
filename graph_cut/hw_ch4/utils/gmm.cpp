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
#include "gmm.h"
#include "display.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <random>
#include <set>

//! distribution need to change a interface, vector to cv::Mat
static std::random_device rd;
static std::mt19937 rng(rd());

std::set<int> get_random_index(int max_idx, int n);

/*--------------------------------------------------------
#####################implementation: Gaussian3D #####################
---------------------------------------------------------*/
Gaussian3D::Gaussian3D(const cv::Matx31d& miu, const cv::Matx33d& sigma)
    : miu_(miu), sigma_(sigma) {
}

double Gaussian3D::compute_gaussian_pdf(const cv::Matx31d& sample) {
    double coeff =
        cv::pow(M_1_PI * 0.5, 3 / 2) / std::sqrt(cv::determinant(sigma_));
    auto tmp = -0.5 * (sample - miu_).t() * sigma_.inv() * (sample - miu_);
    return coeff * exp(tmp(0));
}

cv::Mat Gaussian3D::compute_dataset_gaussian_pdf_map(cv::Mat samples) {
    cv::Mat result = cv::Mat::zeros(cv::Size(1, samples.rows), CV_64F);
    for (int i = 0; i < samples.rows; i++) {
        result.at<double>(i) = compute_gaussian_pdf(samples.at<cv::Vec3d>(i));
    }
    return result;
}
cv::Mat Gaussian3D::compute_gaussian_pdf_map(cv::Mat img) {
    cv::Mat result = cv::Mat::zeros(img.size(), CV_64FC1);
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            result.at<double>(r, c) =
                compute_gaussian_pdf(img.at<cv::Vec3b>(r, c));
        }
    }
    return result;
}
cv::Matx31d Gaussian3D::get_miu() const {
    return miu_;
}

cv::Matx33d Gaussian3D::get_sigma() const {
    return sigma_;
}

void Gaussian3D::set_miu(const cv::Matx31d& miu) {
    miu_ = miu;
}

void Gaussian3D::set_sigma(const cv::Matx33d& sigma) {
    sigma_ = sigma;
}
/*--------------------------------------------------------
#####################implementation: GMM #####################
---------------------------------------------------------*/
GMM::GMM(cv::Mat img, int num_gaussian)
    : EMBase(),
      img_(img.clone()),
      samples_(img.reshape(1, img.rows * img.cols)),
      w_gaussian_model_(num_gaussian, 1.0 / num_gaussian),
      gaussian3d_model_(num_gaussian),
      posterior_(num_gaussian,
                 cv::Mat::zeros(cv::Size(1, img.rows * img.cols), CV_64FC1)) {
    samples_.convertTo(samples_, CV_64FC1);
    mode_ = GMM_MODE::IMAGE_MODE;
}

GMM::GMM(cv::Mat img, const std::vector<cv::Point>& scribble, int num_gaussian)
    : EMBase(),
      img_(img.clone()),
      samples_(cv::Mat::zeros(cv::Size(3, scribble.size()), CV_64FC1)),
      w_gaussian_model_(num_gaussian, 1.0 / num_gaussian),
      gaussian3d_model_(num_gaussian),
      posterior_(num_gaussian,
                 cv::Mat::zeros(cv::Size(1, scribble.size()), CV_64FC1)) {
    for (int i = 0; i < scribble.size(); i++) {
        samples_.at<cv::Vec3d>(i) = img_.at<cv::Vec3b>(scribble[i]);
    }
    mode_ = GMM_MODE::SCRIBBLE_MODE;
}

std::set<int> get_random_index(int max_idx, int n) {
    std::uniform_int_distribution<int> dist(1, max_idx + 1);
    std::set<int> random_idx;
    while (random_idx.size() < n) {
        random_idx.insert(dist(rng) - 1);
    }
    return random_idx;
}

void GMM::initialize() {
    std::set<int> random_idx =
        get_random_index(samples_.rows - 1, gaussian3d_model_.size());

    for (auto it = random_idx.begin(); it != random_idx.end(); it++) {
        cv::Matx31d miu = samples_.at<cv::Vec3d>(*it);
        cv::Matx33d sigma_square = 250 * 250 * cv::Matx33d::eye();

        gaussian3d_model_[std::distance(random_idx.begin(), it)].set_miu(miu);
        gaussian3d_model_[std::distance(random_idx.begin(), it)].set_sigma(
            sigma_square);
    }
}

void GMM::update_e_step() {
    cv::Mat sum = cv::Mat::zeros(posterior_[0].size(), posterior_[0].type());
    for (int i = 0; i < gaussian3d_model_.size(); i++) {
        posterior_[i] =
            gaussian3d_model_[i].compute_dataset_gaussian_pdf_map(samples_);
        sum += w_gaussian_model_[i] * posterior_[i];
    }

    double sum_posterior = 0.0;
    for (int i = 0; i < gaussian3d_model_.size(); i++) {
        cv::divide(posterior_[i], sum, posterior_[i]);
        sum_posterior += w_gaussian_model_[i] * cv::sum(posterior_[i])[0];
    }
    // std::cerr << "sum_posterior :" << sum_posterior / (img_.rows * img_.cols)
    //           << '\n';
}

void GMM::update_m_step() {
    for (int id_model = 0; id_model < gaussian3d_model_.size(); id_model++) {
        double Nk = cv::sum(posterior_[id_model])[0];
        update_miu(id_model, Nk);
        update_sigma(id_model, Nk);
        update_weight(id_model, Nk);
    }
}

void GMM::update_miu(int id_model, double Nk) {
    cv::Matx31d new_miu = cv::Matx31d::zeros();
    for (int i = 0; i < samples_.rows; i++) {
        new_miu +=
            posterior_[id_model].at<double>(i) * samples_.at<cv::Vec3d>(i);
    }

    //gaussian3d_model_[id_model].set_miu(new_miu.mul(1 / (Nk + 1e-10)));
    gaussian3d_model_[id_model].set_miu((1 / (Nk + 1e-10)) * new_miu);
}

void GMM::update_sigma(int id_model, double Nk) {
    cv::Matx33d new_sigma = cv::Matx33d::zeros();
    for (int i = 0; i < samples_.rows; i++) {
        cv::Matx31d resi = cv::Matx31d(samples_.at<cv::Vec3d>(i)) -
                           gaussian3d_model_[id_model].get_miu();
        new_sigma += posterior_[id_model].at<double>(i) * resi * resi.t();
    }
    new_sigma *= (1 / Nk + 1e-10);
    assert(cv::determinant(new_sigma) > 1e-10);
    gaussian3d_model_[id_model].set_sigma(new_sigma);
}

void GMM::update_weight(int id_model, double Nk) {
    w_gaussian_model_[id_model] = Nk / (samples_.rows);
}

cv::Mat GMM::get_posterior(int id_model) {
    cv::Mat result;
    if (mode_ == GMM_MODE::SCRIBBLE_MODE) {
        cv::Mat img_samples = img_.reshape(1, img_.cols * img_.rows);
        result = gaussian3d_model_[id_model]
                     .compute_dataset_gaussian_pdf_map(img_samples)
                     .reshape(1, img_.rows);

    } else {
        result = posterior_[id_model].reshape(1, img_.rows);
    }

    return result;
}

cv::Mat GMM::get_prob() {
    cv::Mat result = cv::Mat::zeros(img_.size(), CV_64FC1);
    for (int i = 0; i < gaussian3d_model_.size(); i++) {
        result += w_gaussian_model_[i] *
                  gaussian3d_model_[i].compute_gaussian_pdf_map(img_);
    }
    return result;
}