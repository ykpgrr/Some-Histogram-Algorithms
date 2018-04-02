//
//  main.cpp
//  Histogram_Homework
//
//  Created by Yakup Gorur (040130052) on 3/31/18.
//  Copyright © 2018 Yakup Gorur. All rights reserved.
//  Digital Signal Processing Design and Application 2017-2018 Spring
//  Homework 4
//  Lecturer: Prof. Dr. Bilge Gunsel, Research Assistant Yağmur Sabucu

// !!! Please ensure that the image folder path is true.


#include <iostream>
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>       /* pow */
#include <string>



using namespace std; //namespace for  C++ standard library
using namespace cv;  //namespace for  OpenCV library

// !!! Please ensure that the image folder path is true.
// !!!The program writen for "original.jpg" and "distorted.jpg" files

//**************************-User Defined Settings-****************************
//image folder path
string image_folder_path = "/Users/yakup/Software Developer/DigSignalProcess/HW4/";

//To make debug: It shows all outputs in the operations
bool debug = false;

//Histogram Level
#define LEVEL 256


//***********************************--End--***********************************

//quantize function to reduce bits
void Quantize_Function(const cv::Mat &input, cv::Mat &output, size_t div);
void Quantize_Function_with_K_Means(const cv::Mat &input, cv::Mat &output, int K);

//Histogram Equalization
void EqualizeHist_Function(const Mat1b& src, Mat1b& dst);

//Calculate the Histogram
void Image2Histogram(Mat image, float histogram[]);

//Calculate the Cumulative Histogram
void Histogram2CumulativeHistogram(float histogram[], float cumulativeHistogram[]);

//Histogram Matching
void HistogramMatching(const Mat& inputImage, const Mat& desiredImage, Mat& outputImage);

//Show Histogram
void showHistogram(const Mat& image, string fileName);
void showHistogram16bin(const Mat& image, string fileName);


int main(int argc, const char * argv[]) {
    
    //Reading the "original.jpg" image file into original Mat Object
    Mat original = imread(image_folder_path + "original.jpg");
    
    //Check the Is the file opened or not?
    if (original.empty()){
        std::cerr<<"can't open image"<<std::endl;
        return - 1;
    }
    
    //Debug: Show the original image file
    if(debug){ imshow("original image file", original); waitKey();}
    
    Mat original_gray(original.size(), CV_8UC1); //Original_Gray Mat object 8 bits
    
    //Convert to original image to 8bits gray level image.
    cvtColor(original, original_gray, COLOR_BGR2GRAY);
    
    //Debug: Show the gray level image
    if(debug){ imshow("Gray Level Image File", original_gray); waitKey();}
    
    //Write the 8bits gray level image as original_gray.jpg
    imwrite(image_folder_path + "original_gray.jpg", original_gray);
    
    //Reading the original_gray.jpg file
    original_gray = imread(image_folder_path + "original_gray.jpg", IMREAD_GRAYSCALE);
    
    //Check the Is the file opened or not?
    if (original_gray.empty()){
        std::cerr<<"can't open image"<<std::endl;
        return - 1;
    }
    
    
    //Quantize the 8bits gray level image to 4bits gray level image with K-means
    Mat original_gray_4_Kmeans(original_gray.size(), CV_8UC1);
    Quantize_Function_with_K_Means(original_gray, original_gray_4_Kmeans, 16);
    if(debug){ imshow("Quantized Gray Level Image with K means", original_gray_4_Kmeans); waitKey();}
    imwrite(image_folder_path + "original_gray_4_Kmeans.jpg", original_gray_4_Kmeans);
    
    /*
    cout<<"\n";
    for (int y = 0; y < original_gray_4_Kmeans.rows; y++){
        for (int x = 0; x < original_gray_4_Kmeans.cols; x++){
            cout<< (int) ( original_gray_4_Kmeans.at<uchar>(y, x) )<<" ";
        }
    cout<<"\n";
    }
    */
    
    //Quantize the 8bits gray level image to 4bits gray level image.
    Mat original_gray_4_lut(original_gray.size(), CV_8UC1); //Original_Gray_4 Mat object
    Quantize_Function(original_gray, original_gray_4_lut, pow(2,4));
    if(debug){ imshow("Quantized Gray Level Image with function", original_gray_4_lut); waitKey();}
    imwrite(image_folder_path + "original_gray_4_lut.jpg", original_gray_4_lut);
    


    //Quantize the 8bits gray level image to 4bits gray level image with basic method
    uchar N = 16;
    Mat original_gray_4_basic(original_gray.size(), CV_8UC1);
    original_gray_4_basic = original_gray / N;
    original_gray_4_basic = original_gray_4_basic * N;
    //Debug: Show the quantized gray level image
    if(debug){ imshow("Quantized Gray Level Image with basic method", original_gray_4_basic); waitKey();}
    
    //Write the quantized gray level image as original_gray_4.jpg
    imwrite(image_folder_path + "original_gray_4_withbasic.jpg", original_gray_4_basic);
    
    /*
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(50);
    compression_params.push_back(IMWRITE_JPEG_OPTIMIZE);
    compression_params.push_back(1);
    compression_params.push_back(IMWRITE_JPEG_PROGRESSIVE);
    compression_params.push_back(1);
    imwrite(image_folder_path + "original_gray_4_withbasic.jpg", original_gray_4_basic, compression_params);
     */
    
    showHistogram(original_gray_4_basic, "Large_Scale_original_gray_4_Histogram");
    showHistogram16bin(original_gray_4_basic, "Low_Scale_original_gray_4_Histogram");
    waitKey();

    
    
    //reading distorted.jpg image file
    Mat distorted = imread(image_folder_path + "distorted.jpg");
    
    //Check the Is the file opened or not?
    if (distorted.empty()){
        std::cerr<<"can't open distorted image"<<std::endl;
        return - 1;
    }
    
    //Debug: Show the distorted image file
    if(debug){ imshow("distorted image file", distorted); waitKey();}
    
    //Convert to distorted image to 8bits gray level image.
    //Mat1b distorted_gray(distorted.rows, distorted.cols); //Distorted_Gray Mat object 8 bits
    Mat distorted_gray;
    cvtColor(distorted, distorted_gray, COLOR_BGR2GRAY);
    
    //Debug: Distorted Gray Image File
    if(debug){ imshow("Distorted Gray Image File", distorted_gray); waitKey();}
    
    //Write the 8bits gray level image as distorted_gray.jpg
    imwrite(image_folder_path + "distorted_gray.jpg", distorted_gray);
    
    
    //Histogram Equation
    Mat1b histeq_distorted_gray;
    EqualizeHist_Function(distorted_gray, histeq_distorted_gray);
    //Debug: Show Histogrom Equalized Distorted Gray Image File
    if(debug){ imshow("Histogrom Equalized Distorted Gray Image File", histeq_distorted_gray); waitKey();}
    
    //Write Image
    imwrite(image_folder_path + "histeq_distorted_gray.jpg", histeq_distorted_gray);
    
    //Histogram Matching
    Mat histmatch_distorted_gray; //Mat Object for output of histogram matching function
    HistogramMatching(distorted_gray, original_gray, histmatch_distorted_gray);
    
    //Debug: Show the gray level image
    if(debug){ imshow("Histogrom Matching Image -> between Distorted Gray and Original Gray", histmatch_distorted_gray); waitKey();}
    
    //Writing the histogram matching image file as histmatch_distorted_gray.jpg
    imwrite(image_folder_path + "histmatch_distorted_gray.jpg", histmatch_distorted_gray);
    
    //Showing Histogram
    imshow("Distorted Gray (Original) Image", distorted_gray);
    showHistogram(distorted_gray, " Distorted Gray (Original) Histogram");
    
    imshow("Original Gray (Desired) Image", original_gray);
    showHistogram(original_gray, "Original Gray (Desired) Image Histogram");
    
    imshow("Histogram Matched Image", histmatch_distorted_gray);
    showHistogram(histmatch_distorted_gray, "Histogram Matched Image Histogram");
    
    waitKey();
    
    return 0;
}



