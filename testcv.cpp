#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;

int main( int argc, char** argv )
{
 
    std::string img = "lenna.jpg";
    Mat srcImage = imread(img, IMREAD_GRAYSCALE);
    namedWindow("Show_Image");

    imshow("Show_Image", srcImage);
    waitKey(0);
    return 0;
}