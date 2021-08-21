#include "interaction_tool.h"
#include "opencv_utils.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void drag_to_collect_pixel(int event, int x, int y, int flags, void* ptr) {
    CallbackItem* item_ptr = reinterpret_cast<CallbackItem*>(ptr);
    if ((flags & cv::EVENT_FLAG_LBUTTON) && event == cv::EVENT_MOUSEMOVE &&
        is_in_img(item_ptr->img_, y, x)) {
        item_ptr->points_.emplace_back(x, y);
        cv::circle(item_ptr->img_, cv::Point(x, y), 1, item_ptr->color_, 2);

        cv::imshow(item_ptr->win_name_, item_ptr->img_);
        cv::waitKey(1);
    }
}

std::array<std::vector<cv::Point>, 2> drag_to_get_fore_and_background_scrible(
    cv::Mat img_origin) {
    cv::Mat img = img_origin.clone();

    cv::Scalar color_foreground(0, 255, 0);
    std::string win_name =
        "please drag to collect foreground, then press any key to continue";
    CallbackItem item_foregroud(img, win_name, color_foreground);
    // collect foregound scribles;
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(win_name, drag_to_collect_pixel,
                         (void*)&item_foregroud);
    cv::imshow(win_name, img);
    cv::waitKey(0);
    cv::destroyWindow(win_name);
    // collect background scribles;
    cv::Scalar color_background(0, 0, 255);
    win_name = "please drag to collect background";
    CallbackItem item_backgroud(item_foregroud.img_, win_name,
                                color_background);
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(win_name, drag_to_collect_pixel,
                         (void*)&item_backgroud);
    cv::imshow(win_name, img);
    cv::waitKey(0);

    return {item_foregroud.points_, item_backgroud.points_};
}

ScribbleInteractionTool::ScribbleInteractionTool(cv::Mat img)
    : marked_points_(drag_to_get_fore_and_background_scrible(img)) {
}