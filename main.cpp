#include <opencv2/opencv.hpp>
#include <string>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::imshow("Display window", image);
    cv::waitKey(0);
    return 0;
}