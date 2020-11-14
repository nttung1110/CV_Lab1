#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
//https://riptutorial.com/opencv/example/23407/adjusting-brightness-and-contrast-of-an-image-in-cplusplus

// int brightnessRatio=50;

void CallbackFunctionGaus (int kernel_size, void *userData){
    std::string imgFilename = *( static_cast < std::string * >( userData ) );
    Mat srcImage = imread(imgFilename, IMREAD_COLOR);
    if(kernel_size == 0) return ;

    kernel_size = kernel_size*2+1;
    // kernel size must be odd
    int half = kernel_size/2;

    int srcRow = srcImage.rows;
    int srcCol = srcImage.cols;
    int tarRow = srcRow-kernel_size+1;
    int tarCol = srcCol-kernel_size+1;
    int nChannel = srcImage.channels();
    Mat tmpImage = srcImage;
    Mat targImage = Mat(tarRow, tarCol, srcImage.type());

    double GausKernel[kernel_size][kernel_size];
    // generate gaussian filter

    double sigma = 5;
    double s = 2.0 * sigma * sigma; 

    double sum = 0;
    for (int x = -half; x <= half; x++) { 
        for (int y = -half; y <= half; y++) { 
            double r = x * x + y * y; 
            GausKernel[x+half][y+half] = (exp(-r / s)) / (3.14 * s); 
            sum += GausKernel[x+half][y+half]; 
        } 
    }

    // normalize kernel
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j){
            GausKernel[i][j] /= sum;  
        }
    }

    // filtering
    for(int x=0; x<tarRow; x++){
        for(int y=0; y<tarCol; y++){
            int start_x = x;
            int start_y = y;
            int end_x = start_x + kernel_size - 1;
            int end_y = start_y + kernel_size - 1;

            // start filtering

            double sum_b = 0.0;
            double sum_g = 0.0;
            double sum_r = 0.0;
            
            for(int source_x=start_x; source_x<=end_x; source_x+=1){
                for(int source_y=start_y; source_y<end_y; source_y+=1){
                    int pos_x_gaus_fil = source_x - start_x;
                    int pos_y_gaus_fil = source_y - start_y;

                    double gaus_coef = GausKernel[pos_x_gaus_fil][pos_y_gaus_fil];

                    sum_b += gaus_coef*tmpImage.at<Vec3b>(source_x, source_y)[0];
                    sum_g += gaus_coef*tmpImage.at<Vec3b>(source_x, source_y)[1];
                    sum_r += gaus_coef*tmpImage.at<Vec3b>(source_x, source_y)[2];
                }
            }

            // assign value
            targImage.at<Vec3b>(x, y)[0] = sum_b;
            targImage.at<Vec3b>(x, y)[1] = sum_g;
            targImage.at<Vec3b>(x, y)[2] = sum_r;
        }
    }

    imshow ("Source_Image" , targImage);
}

void CallbackFunctionAvg (int kernel_size , void * userData )
{
    // cast userData to value
    std::string imgFilename = *( static_cast < std::string * >( userData ) );
    Mat srcImage = imread(imgFilename, IMREAD_COLOR);

    if(kernel_size == 0) return ;

    int srcRow = srcImage.rows;
    int srcCol = srcImage.cols;


    int tarRow = srcRow-kernel_size+1;
    int tarCol = srcCol-kernel_size+1;

    int nChannel = srcImage.channels();

    // Initialize accumulate matrix with default value as 0
	std::vector<std::vector<int>> accumMatrixB(srcRow, std::vector<int>(srcCol, 0));
	std::vector<std::vector<int>> accumMatrixG(srcRow, std::vector<int>(srcCol, 0));
	std::vector<std::vector<int>> accumMatrixR(srcRow, std::vector<int>(srcCol, 0));
    // Building accumMatrix such that when we access accumMatrix[i][j] we can get 
    // the sum of all elements of sub matrix from [0][0] to [i][j]
    
    uint8_t* pData = (uint8_t*)srcImage.data;
    for(int x=0; x<srcRow; x++){
        for(int y=0; y<srcCol; y++){
            if(x==0 || y==0){
                accumMatrixB[x][y] = srcImage.at<Vec3b>(x, y)[0];
                accumMatrixG[x][y] = srcImage.at<Vec3b>(x, y)[1];
                accumMatrixR[x][y] = srcImage.at<Vec3b>(x, y)[2]; 
            }
            else{
                
                accumMatrixB[x][y] = accumMatrixB[x-1][y] + accumMatrixB[x][y-1] - accumMatrixB[x-1][y-1] + srcImage.at<Vec3b>(x, y)[0];
                accumMatrixG[x][y] = accumMatrixG[x-1][y] + accumMatrixG[x][y-1] - accumMatrixG[x-1][y-1] + srcImage.at<Vec3b>(x, y)[1];
                accumMatrixR[x][y] = accumMatrixR[x-1][y] + accumMatrixR[x][y-1] - accumMatrixR[x-1][y-1] + srcImage.at<Vec3b>(x, y)[2];
            }
        }
    }
    Mat targImage = Mat(tarRow, tarCol, srcImage.type());

    // Mat targImage = srcImage;

    // Resolve the boders first
    //  ******
    //  *-----
    //  *-----
    //  *-----
    //  *-----
    //  *-----

    targImage.at<Vec3b>(0, 0)[0] = accumMatrixB[kernel_size-1][kernel_size-1]/(kernel_size*kernel_size); 
    targImage.at<Vec3b>(0, 0)[1] = accumMatrixG[kernel_size-1][kernel_size-1]/(kernel_size*kernel_size);
    targImage.at<Vec3b>(0, 0)[2] = accumMatrixR[kernel_size-1][kernel_size-1]/(kernel_size*kernel_size);
    for(int x=1; x<tarRow; x++){
        targImage.at<Vec3b>(x, 0)[0] = (accumMatrixB[x+kernel_size-1][kernel_size-1] - accumMatrixB[x-1][kernel_size-1])/(kernel_size*kernel_size);
        targImage.at<Vec3b>(x, 0)[1] = (accumMatrixG[x+kernel_size-1][kernel_size-1] - accumMatrixG[x-1][kernel_size-1])/(kernel_size*kernel_size);
        targImage.at<Vec3b>(x, 0)[2] = (accumMatrixR[x+kernel_size-1][kernel_size-1] - accumMatrixR[x-1][kernel_size-1])/(kernel_size*kernel_size);
    }
    for(int y=1; y<tarCol; y++){
        targImage.at<Vec3b>(0, y)[0] = (accumMatrixB[kernel_size-1][y+kernel_size-1] - accumMatrixB[kernel_size-1][y-1])/(kernel_size*kernel_size);
        targImage.at<Vec3b>(0, y)[1] = (accumMatrixG[kernel_size-1][y+kernel_size-1] - accumMatrixG[kernel_size-1][y-1])/(kernel_size*kernel_size);
        targImage.at<Vec3b>(0, y)[2] = (accumMatrixR[kernel_size-1][y+kernel_size-1] - accumMatrixR[kernel_size-1][y-1])/(kernel_size*kernel_size);
    }

    // Then resolve the remaining
    // ------
    // -*****
    // -*****
    // -*****
    // -*****
    // -*****

    for(int start_x=1; start_x<tarRow; start_x++){
        for(int start_y=1; start_y<tarCol; start_y++){
            int end_x = start_x + kernel_size - 1;
            int end_y = start_y + kernel_size - 1;

            Vec3b &intensity = targImage.at<Vec3b>(start_x, start_y);
            intensity.val[0] = (accumMatrixB[start_x+kernel_size-1][start_y+kernel_size-1] 
                                                        - accumMatrixB[start_x-1][start_y+kernel_size-1] - accumMatrixB[start_x+kernel_size-1][start_y-1]
                                                        + accumMatrixB[start_x-1][start_y-1])/(kernel_size*kernel_size); //b

            intensity.val[1] = (accumMatrixG[start_x+kernel_size-1][start_y+kernel_size-1] 
                                                        - accumMatrixG[start_x-1][start_y+kernel_size-1] - accumMatrixG[start_x+kernel_size-1][start_y-1]
                                                        + accumMatrixG[start_x-1][start_y-1])/(kernel_size*kernel_size); //g

            intensity.val[2] = (accumMatrixR[start_x+kernel_size-1][start_y+kernel_size-1] 
                                                        - accumMatrixR[start_x-1][start_y+kernel_size-1] - accumMatrixR[start_x+kernel_size-1][start_y-1]
                                                        + accumMatrixR[start_x-1][start_y-1])/(kernel_size*kernel_size); //r
        }
    }
    imshow ("Source_Image" , targImage);
}

void CallbackFunctionContrast (int pos , void * userData )
{
    std::string imgFilename = *( static_cast < std::string * >( userData ) );
    Mat srcImage = imread(imgFilename, IMREAD_COLOR);

    float ratio = float(pos)/100;
    // first method
    // Mat targImage;
    // srcImage.convertTo(targImage, -1, ratio, 0); 
    
    // second method
    Mat targImage = srcImage;
    int srcRow = srcImage.rows;
    int srcCol = srcImage.cols;


    for(int x=0; x<srcRow; x++){
        for(int y=0; y<srcCol; y++){
            bool check = true;

            int tmp_b = int(targImage.at<Vec3b>(x, y)[0]);
            int tmp_g = int(targImage.at<Vec3b>(x, y)[1]);
            int tmp_r = int(targImage.at<Vec3b>(x, y)[2]);
            
            tmp_b *= ratio;
            tmp_g *= ratio;
            tmp_r *= ratio;

            tmp_b = max(0, min(tmp_b, 255));
            tmp_g = max(0, min(tmp_g, 255));
            tmp_r = min(tmp_r, 255);
            tmp_r = max(0, min(tmp_r, 255));
            targImage.at<Vec3b>(x, y)[0] = tmp_b;
            targImage.at<Vec3b>(x, y)[1] = tmp_g;
            targImage.at<Vec3b>(x, y)[2] = tmp_r;
        }
    }
    imshow ("Source_Image" , targImage);
}

void CallbackFunctionBrightness (int pos , void * userData )
{
    // cast userData to value
    std::string imgFilename = *( static_cast < std::string * >( userData ) );
    Mat srcImage = imread(imgFilename, IMREAD_COLOR);
    
    pos -= 50;

    // first method
    // float ratio = pos/100;
    // Mat targImage;
    // srcImage.convertTo(targImage, -1, 1, pos); //increase the brightness by 20 for each pixel
    
    // second method
    Mat targImage = srcImage;
    int srcRow = srcImage.rows;
    int srcCol = srcImage.cols;

    int my_min = INT_MAX;
    int my_max = -INT_MAX;

    for(int x=0; x<srcRow; x++){
        for(int y=0; y<srcCol; y++){
            bool check = true;

            int tmp_b = int(targImage.at<Vec3b>(x, y)[0]);
            int tmp_g = int(targImage.at<Vec3b>(x, y)[1]);
            int tmp_r = int(targImage.at<Vec3b>(x, y)[2]);
            
            tmp_b += pos;
            tmp_g += pos;
            tmp_r += pos;

            tmp_b = max(0, min(tmp_b, 255));
            tmp_g = max(0, min(tmp_g, 255));
            tmp_r = min(tmp_r, 255);
            tmp_r = max(0, min(tmp_r, 255));
            targImage.at<Vec3b>(x, y)[0] = tmp_b;
            targImage.at<Vec3b>(x, y)[1] = tmp_g;
            targImage.at<Vec3b>(x, y)[2] = tmp_r;
        }
    }
    
    imshow ("Source_Image" , targImage);
}

Mat RGB2Gray(Mat srcImage){
    Mat targImage = Mat(srcImage.rows, srcImage.cols, CV_8UC1);
    for(int i = 0; i<targImage.rows; i++){
        for(int j=0; j<targImage.cols; j++){
            targImage.at<uchar>(i, j) = (srcImage.at<Vec3b>(i, j)[0] + srcImage.at<Vec3b>(i, j)[1] + srcImage.at<Vec3b>(i, j)[2])/3;
        }
    }
    return targImage;
}


void Action(std::string imgFileName, std::string option)
{
    Mat srcImage = imread(imgFileName, IMREAD_COLOR);
    Mat targetImage;

    imshow("Source_Image", srcImage);
    
    if(option == "rgb2gray"){
        targetImage = RGB2Gray(srcImage);
        imshow("Target_Image", targetImage);
        waitKey(0);
    }
    else if(option == "brightness"){
        int pos = 50;
        createTrackbar ("Brightness Ratio", "Source_Image", &pos, 100,
        CallbackFunctionBrightness, &imgFileName );

        imshow ("Source_Image" , srcImage);
        // Wait until user press some key
        waitKey (0);
    }
    else if(option == "contrast"){
        int pos = 100;
        createTrackbar ("Contrast Ratio", "Source_Image", &pos, 100,
        CallbackFunctionContrast, &imgFileName );

        imshow ("Source_Image" , srcImage);
        // Wait until user press some key
        waitKey (0);
    }
    else if(option == "avg"){
        int pos = 0;
        createTrackbar ("Kernel size", "Source_Image", &pos, 20,
        CallbackFunctionAvg, &imgFileName );

        imshow ("Source_Image" , srcImage);
        // Wait until user press some key
        waitKey (0);
    }

    else if(option == "gauss"){
        int pos = 0;
        createTrackbar ("Kernel size", "Source_Image", &pos, 20,
        CallbackFunctionGaus, &imgFileName );

        imshow ("Source_Image" , srcImage);
        // Wait until user press some key
        waitKey (0);
    }
    
}

int main( int argc, char** argv )
{

    if (argc < 3){
        std::cerr << "Usage: " << argv[0] << " -option" << " <FileNameInput>" << std::endl;
        return 1;
    }
    std::string option = argv[1];
    std::string imgFileName = argv[2];
    
    Action(imgFileName, option);
    // Action("lenna.jpg", "avg");
    return 0;
}