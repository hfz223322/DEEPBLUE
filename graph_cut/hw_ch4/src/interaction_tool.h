#include <array>
#include <opencv2/core.hpp>

/**
 * @brief this item is used for OpenCV callback object
 *
 */
struct CallbackItem {
    /**
     * @brief Construct a new Callback Item object
     *
     * @param img : original image
     * @param win_name : window name
     * @param color_scribble : the color for visualization
     */
    CallbackItem(cv::Mat img, std::string win_name, cv::Scalar color_scribble)
        : img_(img), win_name_(win_name), color_(color_scribble){};
    std::string win_name_;           // window name
    cv::Mat img_;                    // original image
    std::vector<cv::Point> points_;  // collected points (scribble)
    cv::Scalar color_;               // visulization color
};

/**
 * @brief opencv call back function for collected scribbles
 *
 * @param img : image to collect scribbles
 * @return std::array<std::vector<cv::Point>, 2> : scribbles, 1st: foreground,
 * 2nd: background
 */
std::array<std::vector<cv::Point>, 2> drag_to_get_fore_and_background_scribbles(
    cv::Mat img);

/**
 * @brief Interaction tool to get scribbles
 *
 */
struct ScribbleInteractionTool {
    /**
     * @brief Construct a new Scribble Interaction Tool object
     *
     * @param img
     */
    ScribbleInteractionTool(cv::Mat img);
    /**
     * @brief Get the points foreground
     *
     * @return std::vector<cv::Point>
     */
    std::vector<cv::Point> get_points_foreground() {
        return marked_points_[0];
    };
    /**
     * @brief Get the points background
     *
     * @return std::vector<cv::Point>
     */
    std::vector<cv::Point> get_points_background() {
        return marked_points_[1];
    };

    std::array<std::vector<cv::Point>, 2>
        marked_points_;  // 0 : foreground scribbles,1: background scribbles
};