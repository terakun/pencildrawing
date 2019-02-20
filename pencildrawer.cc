#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "./pencildrawer.h"

PencilDrawer::PencilDrawer():model_hist_(histsize_){
  // omega_[0] = 11.0;
  // omega_[1] = 37.0;
  // omega_[2] = 52.0;
  omega_[0] = 42.0;
  omega_[1] = 29.0;
  omega_[2] = 29.0;
 
  sigma_b_ = 9.0;
  u_a_ = 105.0;
  u_b_ = 225.0;
  mu_d_ = 90.0;
  sigma_d_ = 11.0;
  direction_num_ = 16;
}


void PencilDrawer::operator()(const cv::Mat &src_img,cv::Mat &dst_img){
  img_rows_ = src_img.rows;
  img_cols_ = src_img.cols;
  dim_ = img_rows_*img_cols_;
  cv::cvtColor(src_img,gray_img_,CV_BGR2GRAY);
  gray_img_.convertTo(src_img_,CV_32F,1.0/255);

  line_drawing();
  pencil_texture();
  cv::Mat pencil_tone_img_float;
  pencil_tone_img_.convertTo(pencil_tone_img_float,CV_32F);
  cv::Mat mul_img = line_img_.mul(pencil_tone_img_float);
  mul_img.convertTo(dst_img,CV_8U);
}

void PencilDrawer::line_drawing(){
  cv::Mat med_img;
  cv::medianBlur(src_img_,med_img,5);
  // make gradient image
  cv::Mat grad_x,grad_y;

  cv::Scharr(med_img, grad_x, CV_32FC1, 1, 0);
  cv::Scharr(med_img, grad_y, CV_32FC1, 0, 1);
  cv::Mat magnitude, direction;
  cv::cartToPolar(grad_x, grad_y, magnitude, direction);
  line_img_ = magnitude.clone();

  // convolute image and make direction_num_ images
  std::vector<cv::Mat> conv_filters;
  int filter_size = std::min(img_rows_,img_cols_) / 80;
  // int filter_size = 15;
  std::vector<cv::Mat> conv_imgs;
  cv::Mat label_img = cv::Mat(img_rows_,img_cols_,CV_8U);
  cv::Mat max_img = cv::Mat(img_rows_,img_cols_,CV_32F);
  max_img = cv::Scalar(0);
  for(int i=0;i<direction_num_;++i){
    cv::Mat conv_filter = cv::Mat(filter_size,filter_size,CV_32F);
    conv_filter = cv::Scalar(0);
    double angle = M_PI*i/(direction_num_-1);
    cv::Point end(filter_size/2+filter_size/2*std::cos(angle),filter_size/2*std::sin(angle));
    cv::line(conv_filter,cv::Point(filter_size/2,0),end,cv::Scalar(1),1);
    conv_filter /= cv::countNonZero(conv_filter);
    conv_filters.push_back(conv_filter);

    cv::Mat conv_img;
    cv::filter2D(magnitude,conv_img,-1,conv_filter,cv::Point(filter_size/2,0));

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
  }
  
  // line shaping
  cv::Mat lineshaping_img = cv::Mat(img_rows_,img_cols_,CV_32F);
  lineshaping_img = cv::Scalar(0);
  for(int i=0;i<direction_num_;++i){
    cv::Mat conv_img;
    cv::filter2D(classified_imgs[i],conv_img,-1,conv_filters[i],cv::Point(filter_size/2,0));
    lineshaping_img += conv_img;
  }

  // inverting pixel values and mapping them to [0,1]
  cv::normalize(lineshaping_img,line_img_,0,1,cv::NORM_MINMAX);
  line_img_ = 1.0 - line_img_;

  for(int r=0;r<img_rows_;++r){
    float* img_ptr = line_img_.ptr<float>(r);
    for(int c=0;c<img_cols_;++c){
      img_ptr[c] = std::pow(img_ptr[c],gamma_); // Gamma correction
    }
  }
  cv::imshow("line",line_img_);
  cv::waitKey(0);
}

void PencilDrawer::pencil_texture(){
  make_tone_histogram();
  cv::Mat preprocess_img;
  cv::medianBlur(gray_img_,preprocess_img,5);
  histogram_mathcing(preprocess_img,model_hist_,tone_img_);
  cv::imshow("tone",tone_img_);
  cv::waitKey(0);

  cv::Mat resized_texture_img;
  cv::resize(texture_img_,resized_texture_img,tone_img_.size());

  Eigen::MatrixXd P;
  cv::cv2eigen(resized_texture_img,P);
  P /= 255.0;
  Eigen::SparseMatrix<double> logP(dim_,dim_);
  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      logP.insert(r*img_cols_+c,r*img_cols_+c) = std::log(P(r,c));
    }
  }

  Eigen::MatrixXd J;
  cv::cv2eigen(tone_img_,J);
  J /= 255.0;
  Eigen::VectorXd logJ(dim_);
  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      logJ[r*img_cols_+c] = std::log(J(r,c));
    }
  }

  Eigen::SparseMatrix<double> Dx(dim_,dim_),Dy(dim_,dim_);
  for(int i=0;i<dim_;++i){
    Dx.insert(i,i) = -1;
    Dx.insert(i,(i+1)%img_cols_+i/img_cols_*img_cols_) = 1;
    Dy.insert(i,i) = -1;
    Dy.insert(i,(i+img_cols_)%dim_) = 1;
  }

  Eigen::SparseMatrix<double> A = lambda_*(Dx*Dx.transpose()+Dy*Dy.transpose())+logP.transpose() * logP;
  Eigen::VectorXd b = logP.transpose() * logJ;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,Eigen::Lower|Eigen::Upper> cg;
  cg.setTolerance(0.1);
  cg.compute(A);
  Eigen::VectorXd beta = cg.solve(b);
  std::cout << "#iterations:     " << cg.iterations() << std::endl;
  std::cout << "estimated error: " << cg.error()      << std::endl;

  Eigen::MatrixXd T(img_rows_,img_cols_);
  Eigen::MatrixXd T_white(img_rows_,img_cols_);
  for(int r=0;r<img_rows_;++r){
    for(int c=0;c<img_cols_;++c){
      T(r,c) = std::pow(P(r,c),beta(r*img_cols_+c));
    }
  }

  cv::Mat pencil_tone_img;
  cv::eigen2cv(T,pencil_tone_img);
  pencil_tone_img.convertTo(pencil_tone_img_,CV_8U,255);
  cv::imshow("tone img",pencil_tone_img_);
  cv::imwrite("toneimg.png",pencil_tone_img_);
  cv::waitKey(0);
}

void PencilDrawer::histogram_mathcing(const cv::Mat &src_img,const std::vector<double> &ref_hist,cv::Mat &dst_img){
  const float range[] = { 0, float(histsize_) } ;
  const float* histRange = { range };

  cv::Mat src_hist;
  cv::calcHist( &src_img, 1, 0, cv::Mat(), src_hist, 1, &histsize_, &histRange);
  
  src_hist /= src_img.size().area();

  std::vector<double> src_cdf(histsize_),ref_cdf(histsize_);
  src_cdf[0] = src_hist.at<float>(0,0);
  ref_cdf[0] = ref_hist[0];
  for(int i=1;i<histsize_;++i){
    src_cdf[i] = src_cdf[i-1] + src_hist.at<float>(0,i);
    ref_cdf[i] = ref_cdf[i-1] + ref_hist[i];
  }

  std::vector<int> lookuptable(histsize_);
  for(int i=0;i<histsize_;++i){
    double min_v = 1.0e10;
    int min_j;
    for(int j=0;j<histsize_;++j){
      if(min_v > std::abs(src_cdf[i]-ref_cdf[j])){
        min_v = std::abs(src_cdf[i]-ref_cdf[j]);
        min_j = j;
      }
    }
    lookuptable[i] = min_j;
  }

  std::vector<double> dst_hist(histsize_);
  dst_img = cv::Mat(src_img.size(),src_img.type());
  for(int r=0;r<src_img.rows;++r){
    const uchar *src_img_ptr = src_img.ptr<uchar>(r);
    uchar *dst_img_ptr = dst_img.ptr<uchar>(r);
    for(int c=0;c<src_img.cols;++c){
      dst_img_ptr[c] = lookuptable[src_img_ptr[c]];
      dst_hist[dst_img_ptr[c]]+=1.0;
    }
  }

  double sum = 0;
  for(auto h:dst_hist) sum += h;
  std::ofstream ofs("./imghist.dat");
  for(int i=0;i<histsize_;++i){
    ofs << i << " " << dst_hist[i]/sum << std::endl;
  }
}

void PencilDrawer::make_tone_histogram(){
  std::ofstream ofs("./hist.dat");
  double sum = 0;
  for(int i=0;i<histsize_;++i){
    double p=0;
    if(u_a_ <= i && i <= u_b_){
      p = 1./(u_b_-u_a_);
    }

    model_hist_[i] = omega_[0]*std::exp(-(255-i)/sigma_b_)/sigma_b_
                    +omega_[1]*p
                    +0.1*omega_[2]*std::exp(-(i-mu_d_)*(i-mu_d_)/(2.0*sigma_d_*sigma_d_))/std::sqrt(2.0*M_PI*sigma_d_);
    sum += model_hist_[i];
  }
  int i = 0;
  for(auto &h:model_hist_){
    h /= sum;
    ofs << i++ << " " << h << std::endl;
  }
}
