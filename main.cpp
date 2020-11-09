#include <iostream>
#include "tools.h"


int main() {
    VideoCapture capture;
    Mat frame;
    cv::CascadeClassifier cascade;
    cascade.load("/usr/local/Cellar/opencv/4.3.0_4/share/opencv4/lbpcascades/lbpcascade_frontalface_improved.xml");
    capture.open(0);
    int counter = 0;
    if( capture.isOpened()){
        while(true){
            capture >> frame;
            counter ++;
            if( frame.empty()) {
                cout<<"Pizda";
                break;
            }
            if (counter > 30) {
                cv::Mat frame1 = frame.clone();
                detect_and_cut(frame1, cascade);
            }
            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    } else {
        cout<<"Could not Open Camera";
    }
    return 0;
}


//mkdir build && cd build
//
//cmake -DCMAKE_BUILD_TYPE=Release  -G"Unix Makefiles" ..
//
//make


// cd /Users/marki.lebyak/projects/videoCut && cd build && ./videoCut
