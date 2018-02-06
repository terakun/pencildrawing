#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./pencildrawer.h"

void PencilDrawer::operator()(const cv::Mat &src_img,cv::Mat &dst_img){
  img_rows_ = src_img.rows;
  img_cols_ = src_img.cols;
  cv::Mat gray_img;
  cv::cvtColor(src_img,gray_img,CV_BGR2GRAY);
  gray_img.convertTo(src_img_,CV_32F,1.0/255);

  line_drawing();
  dst_img = line_img_.clone();
}

void PencilDrawer::line_drawing(){
  // make gradient image
  cv::Mat grad_x,grad_y;

  cv::Scharr(src_img_, grad_x, CV_32FC1, 1, 0);
  cv::Scharr(src_img_, grad_y, CV_32FC1, 0, 1);
  cv::Mat magnitude, direction;
  cv::cartToPolar(grad_x, grad_y, magnitude, direction);
  line_img_ = magnitude.clone();

  // convolute image and make direction_num_ images
  std::vector<cv::Mat> conv_filters;
  int filter_size = std::min(img_rows_,img_cols_) / 60;
  std::vector<cv::Mat> conv_imgs;
  cv::Mat label_img = cv::Mat(img_rows_,img_cols_,CV_8U);
  cv::Mat max_img = cv::Mat(img_rows_,img_cols_,CV_32F);
  max_img = cv::Scalar(0);
  for(int i=0;i<direction_num_;++i){
    cv::Mat conv_filter = cv::Mat(filter_size,filter_size,CV_32F);
    conv_filter = cv::Scalar(0);
    double angle = M_PI*i/(2.0*(direction_num_-1));
    cv::Point end(filter_size*std::cos(angle),filter_size*std::sin(angle));
    cv::line(conv_filter,cv::Point(0,0),end,cv::Scalar(1),1);
    conv_filter /= cv::countNonZero(conv_filter);
    conv_filters.push_back(conv_filter);

    cv::imshow("convolution kernel",conv_filter);
    cv::waitKey(0);

    cv::Mat conv_img;
    cv::filter2D(magnitude,conv_img,-1,conv_filter,cv::Point(0,0));

    for(int r=0;r<img_rows_;++r){
      const float *conv_img_ptr = conv_img.ptr<float>(r);
      float *max_img_ptr = max_img.ptr<float>(r);
      uchar *label_img_ptr = label_img.ptr<uchar>(r);
      for(int c=0;c<img_cols_;++c){
        if( max_img_ptr[c] < conv_img_ptr[c] ){
          max_img_ptr[c] = conv_img_ptr[c];
          label_img_ptr[c] = i;
        }
      }
    }
  }

  std::vector<cv::Mat> classified_imgs;
  for(int i=0;i<direction_num_;++i){
    cv::Mat classified_img = cv::Mat(img_rows_,img_cols_,CV_32F);
    for(int r=0;r<img_rows_;++r){
      float* classified_img_ptr = classified_img.ptr<float>(r);
      const float* magnitude_ptr = magnitude.ptr<float>(r);
      const uchar* label_img_ptr = label_img.ptr<uchar>(r);
      for(int c=0;c<img_cols_;++c){
        if(label_img_ptr[c] == i){
          classified_img_ptr[c] = magnitude_ptr[c];
        }else{
          classified_img_ptr[c] = 0;
        }
      }
    }
    classified_imgs.emplace_back(classified_img);
    cv::imshow("classified",classified_img);
    cv::waitKey(0);
  }
  
  // line shaping
  cv::Mat lineshaping_img = cv::Mat(img_rows_,img_cols_,CV_32F);
  lineshaping_img = cv::Scalar(0);
  for(int i=0;i<direction_num_;++i){
    cv::Mat conv_img;
    cv::filter2D(classified_imgs[i],conv_img,-1,conv_filters[i],cv::Point(0,0));
    lineshaping_img += conv_img;
  }

  // inverting pixel values and mapping them to [0,1]
  cv::normalize(lineshaping_img,line_img_,0,1,cv::NORM_MINMAX);
  line_img_ = 1.0 - line_img_;
}

