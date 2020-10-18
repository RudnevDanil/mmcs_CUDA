#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdio.h>

using namespace cv;
using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

__global__ void cudoCalculculate(unsigned char* image,int res){

    int i = 3 * (threadIdx.x + blockIdx.x * blockDim.x);
    int j = 3 * (threadIdx.x + 1 + blockIdx.x  * blockDim.x);
    if (j > res) return;
     image[i] = image[i + 1] = image[i + 2] = (sqrtf(image[j] - image[i])   +
         sqrtf(image[j+1] - image[i+1]) + sqrtf(image[j+2] - image[i+2] ))*20;
    
}


void CPU() {

    Mat image;
    image = cv::imread("pic.jpg", cv::IMREAD_COLOR);   // Read the file CV_LOAD_IMAGE_COLOR
    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return;
    }
    Mat result = image.clone();
    clock_t start = clock();
    for (int i = 0; i < image.rows - 1; i++)
    {
        //pointer to 1st pixel in row
        Vec3b* p = image.ptr<Vec3b>(i);
        Vec3b* p1 = image.ptr<Vec3b>(i + 1);
        Vec3b* p_r = result.ptr<Vec3b>(i);
        for (int j = 0; j < image.cols - 1; j++)
            //for (int ch = 0; ch < 3; ch++)
            p_r[j][0] = p_r[j][1] = p_r[j][2] = sqrt(pow(p[j + 1][0] - p[j][0], 2) +
                pow(p[j + 1][1] - p[j][1], 2) + pow(p[j + 1][2] - p[j][2], 2) +
                pow(p1[j][0] - p[j][0], 2) + pow(p1[j][1] - p[j][1], 2) +
                pow(p1[j][2] - p[j][2], 2)
            ) * 20;
    }
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "CPU time: " << seconds * 1000 << "ms" << endl;
    imwrite("pic2.jpg", result);
    //show image
    namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display window", result);                   // Show our image inside it.
    cv::waitKey(0);// Wait for a keystroke in the window  
}

void GPU() {
    Mat image;
    image = cv::imread("pic.jpg", cv::IMREAD_COLOR);   // Read the file CV_LOAD_IMAGE_COLOR
    if (!image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return;
    }
    unsigned char* imageGray;
    int full_size_image = image.rows * image.cols * 3;

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    CHECK(cudaMalloc(&imageGray, full_size_image));

    CHECK(cudaMemcpy(imageGray, image.data, full_size_image, cudaMemcpyHostToDevice));

    cudaEventRecord(startCUDA, 0);

    cudoCalculculate <<<(full_size_image / 3 + 255) / 256, 256 >>> (imageGray, full_size_image);

    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << 3 * full_size_image * sizeof(float) / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";

    CHECK(cudaMemcpy(image.data, imageGray, full_size_image, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(imageGray));
    imwrite("pic2GPU.jpg", image);
    namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display window", image);                   // Show our image inside it.
    waitKey(0);

}

int main( int argc, char** argv )
{
    GPU();
    CPU();
    return 0;
}
