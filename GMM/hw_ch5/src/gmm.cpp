#include "gmm.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
namespace gmm {

/*--------------------------------------------------------
#####################implementation: ConfigParam #####################
---------------------------------------------------------*/
ConfigParam::ConfigParam(double a, double alpha, double T) : a_(a), alpha_(alpha), T_(T) {
}

/*--------------------------------------------------------
#####################implementation: GaussianParam #####################
---------------------------------------------------------*/
GaussianParam::GaussianParam(double mean, double var, double weight) : mean_(mean), var_(var), weight_(weight) {
}

/*--------------------------------------------------------
#####################implementation: compute_gaussian_pdf #####################
---------------------------------------------------------*/

double compute_gaussian_pdf(gmm::GaussianParam param, double sample) {
    return (1 / (sqrt(2 * M_PI) * param.var_)) * exp(-0.5 * pow((sample - param.mean_) / param.var_, 2));
}

/*--------------------------------------------------------
#####################implementation: ModelParam #####################
---------------------------------------------------------*/
ModelParam::ModelParam(int num_gaussian) : param_(num_gaussian) {
}
bool cmp_priority(const GaussianParam& lhs, const GaussianParam& rhs) {
    return (lhs.weight_ / lhs.var_) > (rhs.weight_ / rhs.var_);
}

bool cmp_weight(const GaussianParam& lhs, const GaussianParam& rhs) {
    return lhs.weight_ > rhs.weight_;
}

void ModelParam::sort_with_priority() {
    std::sort(this->param_.begin(), this->param_.end(), cmp_priority);
}
void ModelParam::sort_with_weight() {
    std::sort(this->param_.begin(), this->param_.end(), cmp_weight);
}

void ModelParam::normalize_weight() {
    double sum_weight = 0.0;

    for (int i = 0; i < param_.size(); i++) {
        sum_weight += param_[i].weight_;
    }

    std::for_each(param_.begin(), param_.end(), [=](GaussianParam& param) { param.weight_ /= sum_weight; });
}

std::ostream& operator<<(std::ostream& os, const ModelParam& model_param) {
    for (int i = 0; i < model_param.param_.size(); i++) {
        const gmm::GaussianParam& param = model_param.param_[i];
        std::cout << "model id : " << i << " mean : " << param.mean_ << " var : " << param.var_
                  << " weight : " << param.weight_ << '\n';
    }
    return os;
}

/*--------------------------------------------------------
#####################implementation: GMM #####################
---------------------------------------------------------*/
GMM::GMM() : model_param_(num_gaussians_) {
}

void GMM::set_config(const ConfigParam& config) {
    config_param_ = config;
}
void GMM::set_num_gaussian(int num) {
    num_gaussians_ = num;
}

ModelParam& GMM::model_param() {
    return this->model_param_;
}

void GMM::add_sample(double sample) {
    model_param_.sort_with_priority();
    int id = get_gm_id(sample);
    if (id == -1) {
        replace_model(sample);
    } else {
        update_gmm(sample, id);
    }
}
bool GMM::is_in_model(double sample, int id_model) const {
    return std::abs(sample - model_param_.param_[id_model].mean_) <
           config_param_.a_ * model_param_.param_[id_model].var_;
}
int GMM::get_gm_id(double sample) {
    // todo please complete this function, get gaussian model id of the given sample
    for (int i = 0; i < model_param_.param_.size(); i++)
    {
        if (is_in_model(sample, i))
            return i;
    }
    return -1;
}
void GMM::replace_model(double sample) {
    // todo please complete this function, remove a gaussian with lowest priority and add sample as a "gaussian model"
    auto pt = model_param_.param_.end() - 1;
    pt->mean_ = sample;
    pt->var_ = 20;
    pt->weight_ = 0.25;
    //model_param_.normalize_weight();
}

void GMM::update_gmm(double sample, int id) {
    // todo please complete update gmm function
    for (int i = 0; i < model_param_.param_.size(); i++)
    {
        if (i == id)
        {
            model_param_.param_[id].weight_ = (1.0 - config_param_.alpha_) * model_param_.param_[id].weight_ + config_param_.alpha_;
            double p = config_param_.alpha_ * compute_gaussian_pdf(model_param_.param_[id], sample);
            model_param_.param_[id].mean_ = (1.0 - p) * model_param_.param_[id].mean_ + p * sample;
            model_param_.param_[id].var_ = sqrt((1.0 - p) * pow(model_param_.param_[id].var_, 2.0) + p * pow(sample - model_param_.param_[id].mean_, 2.0));
        }
        else
            model_param_.param_[id].weight_ = (1.0 - config_param_.alpha_) * model_param_.param_[id].weight_;
    }
    //model_param_.normalize_weight();
}

bool GMM::is_in_foreground(double sample) {
    // todo please complete this function, if the sample is in the foreground
    model_param_.sort_with_weight();
    double sum = 0.0;
    int B;
    for(int i = 0; i < model_param_.param_.size(); i++)
    {
        sum += model_param_.param_[i].weight_;
        if (sum > config_param_.T_)
        {
            B = i;
            break;
        }    
    }
    for (int i = 0; i < B + 1; i++)
    {
        if(is_in_model(sample, i))
            return false;
    }
    return true;
}

int GMM::num_gaussians_ = 2;
ConfigParam GMM::config_param_(2.5, 0.01, 0.5);

}  // namespace gmm
