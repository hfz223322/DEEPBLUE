#include "k_means.h"
#include <algorithm>
#include <vector>
#include<limits>
#include<iostream>

// to generate random number
static std::random_device rd;
static std::mt19937 rng(rd());

/**
 * @brief get_random_index, check_convergence, calc_square_distance are helper
 * functions, you can use it to finish your homework:)
 *
 */

std::set<int> get_random_index(int max_idx, int n);

float check_convergence(const std::vector<Center>& current_centers,
                        const std::vector<Center>& last_centers);

inline float calc_square_distance(const std::array<float, 3>& arr1,
                                  const std::array<float, 3>& arr2);

float get_loss_value(const std::vector<Sample> &samples, const std::vector<Center> &centers);


/**
 * @brief Construct a new Kmeans object
 *
 * @param img : image with 3 channels
 * @param k : wanted number of cluster
 */
Kmeans::Kmeans(cv::Mat img, const int k) {
    centers_.resize(k);
    last_centers_.resize(k);
    samples_.reserve(img.rows * img.cols);

    // save each feature vector into samples
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            std::array<float, 3> tmp_feature;
            for (int channel = 0; channel < 3; channel++) {
                tmp_feature[channel] =
                    static_cast<float>(img.at<cv::Vec3b>(r, c)[channel]);
            }
            samples_.emplace_back(tmp_feature, r, c, -1);
        }
    }
}

/**
 * @brief initialize k centers randomly, using set to ensure there are no
 * repeated elements
 *
 */
// TODO Try to implement a better initialization function
void Kmeans::initialize_centers() {
    /***std::set<int> random_idx =
        get_random_index(samples_.size() - 1, centers_.size());
    int i_center = 0;

    for (auto index : random_idx) {
        centers_[i_center].feature_ = samples_[index].feature_;
        i_center++;
    }***/
    std::set<int> random_idx = get_random_index(samples_.size() - 1, 1);
    centers_[0] .feature_ = samples_[*random_idx.begin()].feature_;
    for(int k = 1; k < centers_.size(); k++)
    {
        std::vector<float> min_distances;
        for(int i = 0; i < samples_.size(); i++)
        {
            std::set<float> distances;
            for(int j = 0; j < k; j++)
            {
                distances.insert(calc_square_distance(samples_[i].feature_, centers_[j].feature_));
            }
            min_distances.push_back(*distances.begin());
        }
        float sum = 0;
        for(int m = 0; m < min_distances.size(); m++)
        {
            sum = sum + min_distances[m];
        }
        min_distances[0] /= sum;
        for(int n = 1; n < min_distances.size(); n++)
        {
            min_distances[n] /= sum;
            min_distances[n] += min_distances[n-1];
        }
        std::uniform_real_distribution<float> dist(0, 1);
        float p = dist(rng);
        for(int l = 0; l < min_distances.size(); l++)
        {
            if(p <= min_distances[l])
            {
                centers_[k].feature_ = samples_[l].feature_;
                break;
            }
        }
    }
}

/**
 * @brief change the label of each sample to the nearst center
 *
 */
void Kmeans::update_labels() {
    for (Sample& sample : samples_) {
        // TODO update labels of each feature
        int min_index;
        float min_value = std::numeric_limits<float>::max();
        float distances;
        for(int i = 0; i < centers_.size(); i++)
        {
            distances = calc_square_distance(sample.feature_, centers_[i].feature_);
            if(distances < min_value)
            {
                min_value = distances;
                min_index = i;
            }
        }
        sample.label_ = min_index;
    }
}           


/**
 * @brief move the centers according to new lables
 *
 */
void Kmeans::update_centers() {
    // backup centers of last iteration
    last_centers_ = centers_;
    // calculate the mean value of feature vectors in each cluster
    // TODO complete update centers functions.
    for(int i = 0; i < centers_.size(); i++)
    {
        std::array<float,3 > sum = {0.0, 0.0, 0.0};
        int nums = 0;
        for(int j = 0; j < samples_.size(); j++)
        {
            if(samples_[j].label_ == i)
            {
                for(int k = 0; k < 3; k++)
                    sum[k] += samples_[j].feature_[k];
                nums++;
            }
        }
        for(int m = 0; m < 3; m++)
            sum[m] /= nums;
        centers_[i].feature_ = sum;
    }
}

/**
 * @brief check terminate conditions, namely maximal iteration is reached or it
 * convergents
 *
 * @param current_iter
 * @param max_iteration
 * @param smallest_convergence_radius
 * @return true
 * @return false
 */
bool Kmeans::is_terminate(int current_iter, int max_iteration,
                          float smallest_convergence_radius) const {
    // TODO Write a terminate function.
    // helper funtion: check_convergence(const std::vector<Center>&
    // current_centers, const std::vector<Center>& last_centers)
    if((current_iter >= max_iteration) || (check_convergence(centers_, last_centers_) <= smallest_convergence_radius))
        return true;
}

std::vector<Sample> Kmeans::get_result_samples() const {
    return samples_;
}
std::vector<Center> Kmeans::get_result_centers() const {
    return centers_;
}
/**
 * @brief Execute k means algorithm
 *                1. initialize k centers randomly
 *                2. assign each feature to the corresponding centers
 *                3. calculate new centers
 *                4. check terminate condition, if it is not fulfilled, return
 *                   to step 2
 * @param max_iteration
 * @param smallest_convergence_radius
 */
void Kmeans::run(int max_iteration, float smallest_convergence_radius) {
    initialize_centers();
    int current_iter = 0;
    while (!is_terminate(current_iter, max_iteration,
                         smallest_convergence_radius)) {
        current_iter++;
        update_labels();
        update_centers();
    }
    std::cout << "kmeans++\n";
    std::cout << "loss value:" << get_loss_value(samples_, centers_) << std::endl;
    std::cout << "iter_times:" << current_iter << std::endl;
}

/**
 * @brief Get n random numbers from 1 to parameter max_idx
 *
 * @param max_idx
 * @param n
 * @return std::set<int> A set of random numbers, which has n elements
 */
std::set<int> get_random_index(int max_idx, int n) {
    std::uniform_int_distribution<int> dist(1, max_idx + 1);

    std::set<int> random_idx;
    while (random_idx.size() < n) {
        random_idx.insert(dist(rng) - 1);
    }
    return random_idx;
}
/**
 * @brief Calculate the L2 norm of current centers and last centers
 *
 * @param current_centers current assigned centers with 3 channels
 * @param last_centers  last assigned centers with 3 channels
 * @return float
 */
float check_convergence(const std::vector<Center>& current_centers,
                        const std::vector<Center>& last_centers) {
    float convergence_radius = 0;
    for (int i_center = 0; i_center < current_centers.size(); i_center++) {
        convergence_radius +=
            calc_square_distance(current_centers[i_center].feature_,
                                 last_centers[i_center].feature_);
    }
    return convergence_radius;
}

/**
 * @brief calculate L2 norm of two arrays
 *
 * @param arr1
 * @param arr2
 * @return float
 */
inline float calc_square_distance(const std::array<float, 3>& arr1,
                                  const std::array<float, 3>& arr2) {
    return std::pow((arr1[0] - arr2[0]), 2) + std::pow((arr1[1] - arr2[1]), 2) +
           std::pow((arr1[2] - arr2[2]), 2);
}

float get_loss_value(const std::vector<Sample> &samples, const std::vector<Center> &centers)
{
    float loss_value = 0.0;
    for(int i = 0; i < centers.size(); i++)
    {
        for(int j = 0; j < samples.size(); j++)
        {
            if(samples[j].label_ == i)
                loss_value += calc_square_distance(samples[j].feature_, centers[i].feature_);
        }
    }
    return loss_value;
}