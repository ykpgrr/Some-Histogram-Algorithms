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

void Quantize_Function(const cv::Mat &input, cv::Mat &output, size_t div){
    
    //Check, Are the two objects same size
    if(input.data != output.data){
        output.create(input.size(), input.type());
    }
    
    uchar buffer[256];
    for(size_t i = 0; i != 256; ++i){
        buffer[i] = i / div * div + div / 2;
    }
    cv::Mat table(1, 256, CV_8U, buffer, sizeof(buffer));
    cv::LUT(input, table, output);
    
    
    
}
void Quantize_Function_with_K_Means(const cv::Mat &input, cv::Mat &output, int K){
    
    //n shows that numbers of pixel
    int n = input.rows * input.cols;
    
    //data 1D image. It's not matrix anymore. It's a vector
    Mat data = input.reshape(0, n);
    
    //Convert to 32 bit float because of kmeans function input type is 32b float.
    data.convertTo(data, CV_32F);
    
    vector<int> labels; //Cluster label
    Mat1f colors; //Output array
    
    
    kmeans(data, K, labels, cv::TermCriteria(), 4, cv::KMEANS_PP_CENTERS, colors);
    
    
    for (int i = 0; i < n; ++i)
    {
        data.at<float>(i,0) = colors(labels[i],0);
    }
    
    output = data.reshape(0, input.rows);
    output.convertTo(output, CV_8UC1);
    
}

void EqualizeHist_Function(const Mat1b& src, Mat1b& dst)
{
    int cnz = countNonZero(src);
    
    dst = src.clone();
    
    // Histogram
    vector<int> hist(256,0);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            hist[src(r, c)]++;
        }
    }
    
    // Cumulative histogram
    float scale = 255.f / float(cnz);
    vector<uchar> lut(256);
    int sum = 0;
    for (int i = 0; i < hist.size(); ++i) {
        sum += hist[i];
        lut[i] = saturate_cast<uchar>(sum * scale);
    }
    
    // Apply equalization
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            dst(r, c) = lut[src(r,c)];
        }
    }
}

void Image2Histogram(Mat image, float histogram[]){
    
    //Calculate the pixel numbers in input image
    int size = image.rows * image.cols;
    
    //Make all elements of the array zero.
    for(int i = 0; i < LEVEL ; i++){
        histogram[i] = 0;
    }
    
    //Calculate Histogram
    for(int y = 0 ; y < image.rows ; y ++){
        for(int x = 0 ; x < image.cols ; x++){
            histogram[(int)image.at<uchar>(y,x)]++;
        }
    }
    
    //Scale
    for(int i = 0; i < LEVEL ; i++){
        histogram[i] = histogram[i]/size;
    }
    
    return;
}

void Histogram2CumulativeHistogram(float histogram[], float cumulativeHistogram[]){
    
    cumulativeHistogram[0] = histogram[0];
    
    //Just Cumulative Summing.
    for(int i = 1 ; i < LEVEL; i++){
        cumulativeHistogram[i] = histogram[i] + cumulativeHistogram[i - 1];
    }
    return;
}

void HistogramMatching(const Mat& inputImage, const Mat& desiredImage, Mat& outputImage){
    
    //Check the images have one channel.
    if(inputImage.channels() != 1 || desiredImage.channels() != 1){
        cerr<<endl<<"HistogramMatching Function Error.The Input Image or Desired Image does not have only one channel"<<endl;
        cerr<<"Input Image Channels: " <<inputImage.channels()<< endl;
        cerr<<"Desired Image Channels: " <<desiredImage.channels()<< endl;
        return;
    }
    
    //Calculate the Histogram and Cumulative Histogram of Input Image
    float inputHistogram[LEVEL], inputHistogramCumulative[LEVEL];
    Image2Histogram(inputImage, inputHistogram);
    Histogram2CumulativeHistogram(inputHistogram, inputHistogramCumulative);
    
    //Calculate the Histogram and Cumulative Histogram of Desired Image
    float desiredHistogram[LEVEL], desiredHistogramCumulative[LEVEL];
    Image2Histogram(desiredImage, desiredHistogram);
    Histogram2CumulativeHistogram(desiredHistogram, desiredHistogramCumulative);
    
    
    //Histogram Matchin Algorithm
    float outputHistogram[LEVEL]; //Output Histogram
    for(int i = 0; i < LEVEL ; i++){
        int j = 0;
        
        do  {
            outputHistogram[i] = j;
            j++;
        }while (inputHistogramCumulative[i] > desiredHistogramCumulative[j]);
    }
    
    //Output Image Create
    outputImage = inputImage.clone();
    for (int y = 0; y < inputImage.rows; y++){
        for (int x = 0; x < inputImage.cols; x++){
            outputImage.at<uchar>(y, x) = (int) ( outputHistogram[inputImage.at<uchar>(y, x)] );
        }
    }
}

//Function to display histogram of an image and to write the historam in the outout file
void showHistogram(const Mat& image, string fileName){
    int bins = 256;             // number of bins
    int nc = image.channels();    // number of channels
    vector<Mat> histogram(nc);       // array for storing the histograms
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = {0,0,0};      // peak value for each histogram
    
    // The rest of the code will be placed here
    for (int i = 0; i < histogram.size(); i++)
        histogram[i] = Mat::zeros(1, bins, CV_32SC1);
    
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            for (int k = 0; k < nc; k++){
                uchar val = nc == 1 ? image.at<uchar>(i,j) : image.at<Vec3b>(i,j)[k];
                histogram[k].at<int>(val) += 1;
            }
        }
    }
    
    for (int i = 0; i < nc; i++){
        for (int j = 0; j < bins-1; j++)
            hmax[i] = histogram[i].at<int>(j) > hmax[i] ? histogram[i].at<int>(j) : hmax[i];
    }
    
    const char* wname[3] = { "Blue", "Green", "Red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };
    
    for (int i = 0; i < nc; i++){
        canvas[i] = Mat::ones(125, bins, CV_8UC3);
        
        for (int j = 0, rows = canvas[i].rows; j < bins-1; j++){
            line(
                 canvas[i],
                 Point(j, rows),
                 Point(j, rows - (histogram[i].at<int>(j) * rows/hmax[i])),
                 nc == 1 ? Scalar(255, 255, 255) : colors[i],
                 1, 8, 0
                 );
        }
        
        imshow(nc == 1 ? fileName : wname[i]+fileName, canvas[i]);
        string name = string(wname[i])+".jpg";
        imwrite(nc == 1 ? image_folder_path +fileName+".jpg" : image_folder_path+ name, canvas[i]);
    }
}

//Function to display histogram of an image and to write the historam in the outout file
void showHistogram16bin(const Mat& image, string fileName){
    int bins = 16;             // number of bins
    int nc = image.channels();    // number of channels
    vector<Mat> histogram(nc);       // array for storing the histograms
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = {0,0,0};      // peak value for each histogram
    
    // The rest of the code will be placed here
    for (int i = 0; i < histogram.size(); i++)
        histogram[i] = Mat::zeros(1, bins, CV_32SC1);
    
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            for (int k = 0; k < nc; k++){
                uchar val = nc == 1 ? image.at<uchar>(i,j) : image.at<Vec3b>(i,j)[k];
                histogram[k].at<int>(val/16) += 1;
            }
        }
    }
    
    for (int i = 0; i < nc; i++){
        for (int j = 0; j < bins-1; j++)
            hmax[i] = histogram[i].at<int>(j) > hmax[i] ? histogram[i].at<int>(j) : hmax[i];
    }
    
    const char* wname[3] = { "Blue", "Green", "Red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };
    
    for (int i = 0; i < nc; i++){
        canvas[i] = Mat::ones(125, bins, CV_8UC3);
        
        for (int j = 0, rows = canvas[i].rows; j < bins-1; j++){
            line(
                 canvas[i],
                 Point(j, rows),
                 Point(j, rows - (histogram[i].at<int>(j) * rows/hmax[i])),
                 nc == 1 ? Scalar(255, 255, 255) : colors[i],
                 1, 8, 0
                 );
            
        }
        
        imshow(nc == 1 ? fileName : wname[i]+fileName, canvas[i]);
        string name = string(wname[i])+".jpg";
        imwrite(nc == 1 ? image_folder_path +fileName+".jpg" : image_folder_path+ name, canvas[i]);
    }
}



