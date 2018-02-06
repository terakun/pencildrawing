#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "./pencildrawer.h"

int main(int argc,char **argv){
  cv::Mat src_img;
  if(argc < 2){
    src_img = cv::imread("./lenna.png");
  }else{
    src_img = cv::imread(argv[1]);
  }
  cv::Mat dst_img;

  PencilDrawer pencildrawer;
  pencildrawer(src_img,dst_img);

  cv::imshow("source image",src_img);
  cv::imshow("destination image",dst_img);
  cv::imwrite("dst.png",dst_img);
  cv::waitKey(0);
}
