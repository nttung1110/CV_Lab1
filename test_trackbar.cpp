#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;

void CallbackFunction (int pos , void * userData )
{
    // cast userData to value
    int valueFromUser = *( static_cast < int * >( userData ) );
}

int main ( int argc , char ** argv )
{
    // Read original image
    Mat src = imread ("lenna.jpg");
    // Create a window
    namedWindow ( " My ␣ Window " , 1);
    int pos = 50;
    int val = 50;
    // Create track bar
    createTrackbar ( " Trackbar ␣ Name " , " My ␣ Window " , & pos , 100 ,
    CallbackFunction , & val );
    imshow ( " My ␣ Window " , src );
    // Wait until user press some key
    waitKey (0);
    return 0;
}