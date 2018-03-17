#ifndef PENCILDRAWER_H
#define PENCILDRAWER_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <array>

class PencilDrawer{
  cv::Mat src_img_,gray_img_;
  cv::Mat line_img_;
  cv::Mat tone_img_;
  cv::Mat pencil_tone_img_;
  cv::Mat texture_img_;

  int direction_num_;
  int img_rows_ , img_cols_ ;
  int dim_;

  static constexpr int model_num_ = 3;
  const int histsize_ = 256;

  std::array<double,model_num_> omega_;
  std::vector<double> model_hist_;

  double sigma_b_,sigma_d_,u_a_,u_b_,mu_d_,Z_;
  double lambda_;

  void make_tone_histogram();
  void histogram_mathcing(const cv::Mat &src_img,const std::vector<double> &ref_hist,cv::Mat &dst_img);
  public:
  PencilDrawer();
  void line_drawing();
  void pencil_texture();
  void set_lambda(double l){ lambda_ = l ; }
  void set_texture(const cv::Mat &t){
    texture_img_ = t.clone();
  }
  void operator()(const cv::Mat &src_img,cv::Mat &dst_img);

};


#endif
