#include "snake.h"
#include "display.h"
#include <cmath>
#include <iostream>

/**
 * @brief Check if contour is valid
 *
 * @param max_x
 * @param max_y
 * @param radius
 * @param center
 * @return true
 * @return false
 */
bool is_valid(int max_x, int max_y, double radius, cv::Point2d center) {
    return radius < std::min(std::min(max_x - center.x, center.x),
                             std::min(max_y - center.y, center.y));
}

ParamSnake::ParamSnake(double alpha, double beta, double step_size)
    : alpha_(alpha), beta_(beta), step_size_(step_size) {
}

/**
 * @brief Construct a new Contour:: Contour object
 *
 * @param max_x : Boundary in x axis of the image
 * @param max_y : Boundary in y axis of the image
 * @param radius : Define a circle contour with a fixed radius
 * @param center : Define a circle contour with a center point
 * @param num_points : The number of contour points
 */
Contour::Contour(int max_x, int max_y, double radius, cv::Point2d center,
                 int num_points)
    : points_(cv::Mat::zeros(cv::Size(2, num_points), CV_64F)) {
    if (!is_valid(max_x, max_y, radius, center)) {
        std::cerr << "Your Contour are out of boundary." << std::endl;
        std::exit(-1);
    }
    for(int i = 0; i < num_points; i++)
    {
        double angle = 2 *M_PI * i / num_points;
        points_.at<double>(i, 0) = center.x + radius * std::sin(angle);
        points_.at<double>(i, 1) = center.y - radius * std::cos(angle);
    }

    // TODO initialize contour Hints: Circle would be easy to implement with
    // angles!
}

Contour::Contour(cv::Mat points) : points_(points.clone()) {
}

Contour::Contour(const Contour& contour) {
    points_ = contour.points_.clone();
}

int Contour::get_num_points() const {
    return points_.rows;
}

cv::Mat Contour::get_points() const {
    return points_.clone();
}

Contour& Contour::operator=(const Contour& contour) {
    points_ = contour.points_.clone();
}
/**
 * @brief Construct a new Snake:: Snake object
 *
 * @param gvf_x : GVF result for dufuse the contour w.r.t. external force in x
 * axis
 * @param gvf_y : GVF result for dufuse the contour w.r.t. external force in y
 * axis
 * @param contour : initialized contour
 * @param param_snake : relavent parameters for snake model
 */
Snake::Snake(cv::Mat original_img, cv::Mat gvf_x, cv::Mat gvf_y,
             Contour contour, ParamSnake param_snake)
    : GradientDescentBase(param_snake.step_size_),
      original_img_(original_img),
      internal_force_matrix_(cv::Mat::zeros(contour.get_num_points(),
                                            contour.get_num_points(), 0)),
      param_snake_(param_snake),
      contour_(contour),
      last_contour_(contour_),
      gvf_x_(gvf_x.clone()),
      gvf_y_(gvf_y.clone()),
      gvf_contour_(cv::Size(2, contour.get_num_points()), CV_64F) {
    cal_internal_force_matrix();
    std::cout << "OK!" << std::endl;
}
/**
 * @brief Overlodaded operator for [], that points_[i] return the i-th conotur
 * point
 *
 * @param i
 * @return cv::Point2d&
 */
cv::Vec2d& Contour::operator[](int i) {
    return points_.at<cv::Vec2d>(i);
}

/*************************************************************************/
/******************FUNCTIONS FOR SNAKE CLASS******************************/
/************************************************************************/
/**
 * @brief : circularly shifts the values in the array input by downshift and
 * right shift elements
 *
 * @param matrix : matrix need to be dealt with
 * @param down_shift : down shift coefficient
 * @param right_shift : right shift coefficient
 * @return cv::Mat
 */
cv::Mat circshift(cv::Mat matrix, int down_shift, int right_shift) {
    down_shift = ((down_shift % matrix.rows) + matrix.rows) % matrix.rows;
    right_shift = ((right_shift % matrix.cols) + matrix.cols) % matrix.cols;
    cv::Mat output = cv::Mat::zeros(matrix.rows, matrix.cols, matrix.type());

    for (int i = 0; i < matrix.rows; i++) {
        int new_row = (i + down_shift) % matrix.rows;
        for (int j = 0; j < matrix.cols; j++) {
            // int new_colum = (j + down_shift) % input.cols;
            int new_column = (j + right_shift) % matrix.cols;
            output.at<double>(new_row, new_column) = matrix.at<double>(i, j);
        }
    }
    return output;
}
// TODO Implement internal force matrix
void Snake::cal_internal_force_matrix() {
    //  build A matrix using helper function circshift
    cv::Mat identity = cv::Mat::eye(contour_.get_num_points(), contour_.get_num_points(), CV_64F);
    //std::cout << identity.type() << std::endl;
     //std::cout << "OK!" << std::endl;
    cv::Mat A = 2 * identity -circshift(identity, 0, 1) - circshift(identity, 1, 0);
     //std::cout << "OK!" << std::endl;
    std::cout << A << std::endl;
    while(1)
    {
        
    }
    // build B matrix using helper function cirshift
    cv::Mat B = circshift(identity, 0, 2) - 4 * circshift(identity, 0, 1) + 6 * identity - 4 * circshift(identity, 1, 0) + circshift(identity, 2, 0);
    //std::cout << B << std::endl;
    // Build internal force matrix w.r.t. the corresponding parameters
    internal_force_matrix_ = identity - (param_snake_.alpha_ * A - param_snake_.beta_ * B);
    //internal_force_matrix_ = identity - (param_snake_.alpha_ * A - param_snake_.beta_ * B);
    //std::cout << internal_force_matrix_.type() << std::endl;
    std::cout << "OK!" <<std::endl;
}

void Snake::initialize() {
    // Already initialize in constructor.
}

void clapping(cv::Vec2d& point, double max_x, double max_y) {
    point[0] = std::min(std::max(0.0, point[0]), max_x - 1);
    point[1] = std::min(std::max(0.0, point[1]), max_y - 1);
}
// TODO implement update function
void Snake::update() {
    display_contour(original_img_, contour_, 0);

    for (int index = 0; index < contour_.get_num_points(); index++) {
        clapping(contour_[index], gvf_x_.cols, gvf_x_.rows);

        gvf_contour_.at<cv::Vec2d>(index) =
            cv::Vec2d(gvf_x_.at<double>(cv::Point2d(contour_[index])),
                      gvf_y_.at<double>(cv::Point2d(contour_[index])));
    }
    cv::Mat gvf_normalized(gvf_contour_.size(), gvf_contour_.type());
    // Normalize function below will be extremly helpful to tune the parameter
    cv::normalize(gvf_contour_, gvf_normalized, -1, 1, cv::NORM_MINMAX);
    cv::Mat internal_force_matrix_invert;
    cv::invert(internal_force_matrix_, internal_force_matrix_invert);
    cv::Mat new_points = internal_force_matrix_invert * (gvf_normalized + contour_.get_points());
    new_points = contour_.get_points() + param_snake_.step_size_ * (new_points - contour_.get_points());
    contour_ = Contour(new_points);
}

void Snake::print_terminate_info() const {
    std::cout << "Snake iteration finished." << std::endl;
}

Contour Snake::get_contour() const {
    return contour_;
}
// TODO implement energy
double Snake::compute_energy() {
    cv::Mat identity = cv::Mat::eye(contour_.get_num_points(), contour_.get_num_points(), CV_64F);
    cv::Mat C = identity - circshift(identity, 1, 0);
    cv::Mat D = 2 * identity -circshift(identity, 0, 1) - circshift(identity, 1, 0);
    cv::Mat curve_1d = C * contour_.get_points();
    cv::pow(curve_1d, 2.0f, curve_1d);
    double energy_1 = param_snake_.alpha_ * (cv::sum(curve_1d)[0]);
    cv::Mat curve_2d = D * contour_.get_points();
    cv::pow(curve_2d, 2.0f, curve_2d);
    double energy_2 = param_snake_.beta_ * (cv::sum(curve_2d)[0]);
    cv::Mat gvf_contour_square;
    cv::pow(gvf_contour_, 2.0f, gvf_contour_square);
    double energy_3 = cv::sum(gvf_contour_square)[0];
    return energy_1 + energy_2 - energy_3;
}

std::string Snake::return_drive_class_name() const {
    return "Snake";
}

void Snake::roll_back_state() {
    contour_ = last_contour_;
}
void Snake::back_up_state() {
    last_contour_ = contour_;
}