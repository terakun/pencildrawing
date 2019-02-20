#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "./pencildrawer.h"

int main(int argc,char **argv){
  if(argc < 5){
    std::cerr << argv[0] << " [image file] [texture file] [lambda] [gamma]" << std::endl;
    return -1;
  }

  cv::Mat src_img = cv::imread(argv[1]);
  cv::Mat texture_img = cv::imread(argv[2],0);
  cv::Mat dst_img;

  PencilDrawer pencildrawer;
  pencildrawer.set_lambda(std::stod(argv[3]));
  pencildrawer.set_gamma(std::stod(argv[4]));
  pencildrawer.set_texture(texture_img);
  pencildrawer(src_img,dst_img);

  cv::imshow("source image",src_img);
  cv::imshow("destination image",dst_img);
  cv::imwrite("dst.png",dst_img);
  cv::waitKey(0);
}
