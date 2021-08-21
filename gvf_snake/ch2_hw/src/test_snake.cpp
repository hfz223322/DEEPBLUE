#include "display.h"
#include "gvf.h"
#include "snake.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    // preprocess the input image
    cv::GaussianBlur(img, img, cv::Size(3, 3), 3, 3);

    cv::Mat grad_original_x, grad_original_y;
    cv::Sobel(img, grad_original_x, CV_64F, 1, 0, 3);
    cv::Sobel(img, grad_original_y, CV_64F, 0, 1, 3);

    // run gvf at first to get the gradient vecotor flow
    double smooth_term = 1e8;
    double step_size = 2e-9;
    ParamGVF param_gvf(smooth_term, step_size);
    GVF gvf(grad_original_x, grad_original_y, param_gvf);

    int max_iteration_gvf = 500;
    gvf.run(max_iteration_gvf);  // parameter: max_iteration
    std::vector<cv::Mat> gvf_result = gvf.get_result_gvf();

    display_gvf(gvf_result[0], gvf_result[1], 0, true);
    std::cout << gvf_result[0].type() << std::endl;

    // run snake
   /** cv::Mat  gvf_x = cv::imread("gvf_x_star.png", cv::IMREAD_GRAYSCALE);
    cv::Mat  gvf_y = cv::imread("gvf_y_star.png", cv::IMREAD_GRAYSCALE);
    std::vector<cv::Mat> gvf_result(2);
    gvf_result[0] = gvf_x.clone();
    gvf_result[1] = gvf_y.clone();
    display_gvf(gvf_result[0], gvf_result[1], 0, true);**/
    int max_x = gvf_result[0].cols;
    int max_y = gvf_result[1].rows;
    double radius = std::min(max_x, max_y) / 3.5f;
    cv::Point2d center(max_x / 2.f, max_y / 2.f);
    int num_points = 200;

    Contour contour(max_x, max_y, radius, center, num_points);
    ParamSnake param_snake(0.1, 0.1, 1.0);
    //ParamSnake param_snake(0, 0, 1e-10);
    //std::cout << "OK!" << std::endl;
    Snake snake_model(img, gvf_result[0], gvf_result[1], contour, param_snake);
    std::cout << "OK!" << std::endl;
    snake_model.run(5000);
    Contour result_contour = snake_model.get_contour();
    display_contour(img, result_contour, 0);

    return 0;
}