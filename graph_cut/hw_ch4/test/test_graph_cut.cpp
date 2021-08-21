#include "display.h"
#include "graph_cut.h"
#include "tictoc.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char** argv) {
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::resize(img, img, cv::Size(100,100));
    GraphCut gc(img);
    tictoc::tic();
    gc.run();
    std::cout << "Graph cut cost" << tictoc::toc() / 60e6 << " min \n";
    cv::Mat fore_img = gc.get_segmentation(SegType::FOREGROUND);
    cv::Mat back_img = gc.get_segmentation(SegType::BACKGROUND);

    disp_image(fore_img, "fore", 0);
    disp_image(back_img, "back", 0);

    return 0;
}