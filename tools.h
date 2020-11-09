#ifndef VIDEOCUT_TOOLS_H
#define VIDEOCUT_TOOLS_H

#endif //VIDEOCUT_TOOLS_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/core/utility.hpp"

using namespace std;
using namespace cv;


cv::Mat sobel(cv::Mat gray){
    cv::Mat edges;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    cv::Mat edges_x, edges_y;
    cv::Mat abs_edges_x, abs_edges_y;
    Sobel(gray, edges_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
    convertScaleAbs( edges_x, abs_edges_x );
    Sobel(gray, edges_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
    convertScaleAbs(edges_y, abs_edges_y);
    addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, edges);

    return edges;
}


cv::Mat canny(cv::Mat src)
{
    cv::Mat detected_edges;

    int edgeThresh = 1;
    int lowThreshold = 250;
    int highThreshold = 750;
    int kernel_size = 5;
    Canny(src, detected_edges, lowThreshold, highThreshold, kernel_size);

    return detected_edges;
}


void method_2(cv::Mat & src, bool useCanny, bool auto_threshold) {

    // Remove shadows
//    imshow("cutted_2", src);
    cv::Mat hsvImg;
    cvtColor(src, hsvImg, cv::COLOR_BGR2HSV);
    cv::Mat channel[3];
    split(hsvImg, channel);
    channel[2] = cv::Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);

    //Merge channels
    merge(channel, 3, hsvImg);
    cv::Mat rgbImg;
    cvtColor(hsvImg, rgbImg, cv::COLOR_HSV2BGR);
//    imshow("cutted_2", rgbImg);


    // Convert to gray and normalize
    cv::Mat gray(rgbImg.rows, src.cols, CV_8UC1);
    cvtColor(rgbImg, gray, cv::COLOR_BGR2GRAY);
    normalize(gray, gray, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//    imshow("cutted_2", gray);


    // Edge detector
    GaussianBlur(gray, gray, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
    cv::Mat edges;
    if(useCanny){
        edges = canny(gray);
    } else {
        //Use Sobel filter and thresholding.
        edges = sobel(gray);
        if (auto_threshold) {
            threshold(edges, edges, 0, 255, cv::THRESH_OTSU);
        }
        threshold(edges, edges, 25, 255, cv::THRESH_BINARY);
    }


    // Dilate
    cv::Mat dilateGrad = edges;
    int dilateType = cv::MORPH_ELLIPSE;
    int dilateSize = 3;
    cv::Mat elementDilate = getStructuringElement(dilateType,
                                                  cv::Size(2*dilateSize + 1, 2*dilateSize+1),
                                                  cv::Point(dilateSize, dilateSize));
    dilate(edges, dilateGrad, elementDilate);


    // Floodfill
    cv::Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(),
              cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    cv::Mat temp;
    floodFilled(cv::Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    floodFilled = temp;
//    imshow("cutted_2", floodFilled);


    // Erode
    int erosionType = cv::MORPH_ELLIPSE;
    int erosionSize = 4;
    cv::Mat erosionElement = getStructuringElement(erosionType,
                                                   cv::Size(2*erosionSize+1, 2*erosionSize+1),
                                                   cv::Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);
//    imshow("cutted_2", floodFilled);


    // Find largest contour
    int largestArea = 0;
    cv::Rect boundingRectangle;
    int largestContourIndex = 0;
    cv::Mat largestContour(src.rows, src.cols, CV_8UC1, cv::Scalar::all(0));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(floodFilled, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    for(int i=0; i<contours.size(); i++) {
        double a = contourArea(contours[i], false);
        if (a > largestArea) {
            largestArea = a;
            largestContourIndex = i;
            boundingRectangle = boundingRect(contours[i]);
        }
    }
    cv::Scalar color(255, 255, 255);
    drawContours(largestContour, contours, largestContourIndex, color, cv::FILLED, 8, hierarchy);
    rectangle(src, boundingRectangle, cv::Scalar(0, 255, 0), 1, 8, 0);
//    imshow("7. Largest Contour", largestContour);

    // Mask original image
    cv::Mat maskedSrc;
    src.copyTo(maskedSrc, largestContour);
    imshow ("Result", maskedSrc);
//    imshow("src boxed", src);
//    cv::waitKey(0);

}

void detect_and_cut (cv::Mat &frame1,  cv::CascadeClassifier &cascade) {
    std::vector<cv::Rect> faces;
    cv::Mat frame_copied = frame1.clone();
    cv::Mat gray, smallImg;
    cv::Mat cropped;
    cv::Scalar color = cv::Scalar(255, 0, 0);
    cvtColor( frame_copied, gray, cv::COLOR_BGR2GRAY );
    resize( gray, smallImg, Size(), 0.75, 0.75, INTER_LINEAR);
    resize(frame_copied, frame_copied, Size(), 0.75, 0.75, INTER_LINEAR);
    equalizeHist( smallImg, smallImg );
    cascade.detectMultiScale(smallImg, faces, 1.1, 3, 0);
    if (faces.empty()){
        cropped = frame_copied;
        std::cout << "Face not detected" << std::endl;
    }else {
        std::cout << "Face detected" << std::endl;
        for (const auto& face : faces) {
            cv::Point point1, point2;
            if (face.x - face.width < 0) {point1 = cv::Point(0, face.y - face.height);
            } else if (face.y - face.height < 0){point1 = cv::Point(face.x - face.width, 0);
            } else if (face.x - face.width < 0 && face.y - face.height < 0) {point1 = cv::Point(0, 0);
            } else {point1 = cv::Point(face.x - face.width, face.y - face.height);}

            if (face.x + 2 * face.width > frame_copied.cols) {point2 = cv::Point(frame_copied.cols, frame_copied.rows);}
            else {point2 = cv::Point(face.x + 2 * face.width, frame_copied.rows);}

            cv::Rect roi(point1, point2);
            cropped = frame_copied(roi);
        }
//        imshow("canvasOutput", cropped);
    }
    method_2(cropped, true, false);
}