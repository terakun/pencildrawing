#ifndef PENCILDRAWER_H
#define PENCILDRAWER_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

class PencilDrawer{
  cv::Mat src_img_;
  cv::Mat line_img_;

  const int direction_num_ = 8;
  int img_rows_ , img_cols_ ;
  public:
  void line_drawing();
  void pencil_texture();
  void operator()(const cv::Mat &src_img,cv::Mat &dst_img);
};


#endif
