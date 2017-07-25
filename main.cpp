#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <calib3d/calib3d.hpp>
#include <imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main()
{
  // Constants
  const cv::Size board_size(8, 6);
  cout << "board_size.width " << board_size.width << endl;
  cout << "board_size.height " << board_size.height << endl;
  const float square_size = 1;  // square size in some user-defined units.

  // Part I: Camera Calibration
  const vector<string> calibration_file_names = {"pic001.jpg", "pic002.jpg", "pic003.jpg", "pic004.jpg", "pic005.jpg",
                                                 "pic006.jpg", "pic007.jpg", "pic008.jpg", "pic009.jpg", "pic010.jpg",
                                                 "pic011.jpg", "pic012.jpg", "pic013.jpg", "pic014.jpg", "pic015.jpg",
                                                 "pic016.jpg", "pic017.jpg", "pic018.jpg"};
  const string calibration_directory = "../Images/Calibration/";

  vector<vector<cv::Point2f>> image_points;
  vector<vector<cv::Point3f>> object_points;
  cv::Mat raw_image;
  for (size_t i = 0; i < calibration_file_names.size(); ++i)
  {
    raw_image = cv::imread(calibration_directory + calibration_file_names[i]);

    // Find chessboard corners:
    vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(raw_image, board_size, corners);
    if (found)
    {
      image_points.push_back(corners);

      vector<cv::Point3f> points;
      for (int i = 0; i < board_size.height; ++i)
      {
        for (int j = 0; j < board_size.width; ++j)
        {
          points.push_back(cv::Point3f(static_cast<float>(j*square_size), static_cast<float>(i*square_size), 0));
        }
      }
      object_points.push_back(points);
    }

    cv::drawChessboardCorners(raw_image, board_size, corners, found);
    cv::imshow("Calibration", raw_image);
    cv::waitKey(100);
  }

  cv::Mat camera_matrix;
  cv::Mat distortion_coeffs;
  vector<cv::Mat> r_vecs;
  vector<cv::Mat> t_vecs;
  double reprojection_error = cv::calibrateCamera(object_points, image_points, raw_image.size(), camera_matrix, distortion_coeffs, r_vecs, t_vecs);
  cout << "reprojection_error " << reprojection_error << endl;
  cv::FileStorage calibration_file("Calibration.json", cv::FileStorage::WRITE);
  calibration_file << "camera_matrix" << camera_matrix;
  calibration_file << "distortion_coeffs" << distortion_coeffs;
  cout << "Parameters are written in Calibration.json" << endl;

  // Part II: Undistort the images
  const string undistorted_directory = "../Images/Undistorted/";
  for (size_t i = 0; i < calibration_file_names.size(); ++i)
  {
    raw_image = cv::imread(calibration_directory + calibration_file_names[i]);

    cv::Mat undistorted_image;
    cv::undistort(raw_image, undistorted_image, camera_matrix, distortion_coeffs);
    cv::imwrite(undistorted_directory + "undistorted_" + calibration_file_names[i], undistorted_image);
  }

  // Part III: 3D position estimation. Assume the objects are flat and lie on the same plane of chessboard in "pic018.jpg"
  cout << "3D position estimation: " << endl;
  const string position_estimation_directory = "../Images/PositionEstimation/";
  const vector<string> position_estimation_file_names = {"obj000.jpg", "obj001.jpg", "obj002.jpg", "obj003.jpg", "obj004.jpg"};

  cv::Mat rotation_matrix;
  cv::Rodrigues(r_vecs.back(), rotation_matrix);
  cv::Mat translation_matrix = t_vecs.back();

  cv::Mat chessboard_image = cv::imread(position_estimation_directory + "checkerboard.jpg");
  cv::Mat undistorted_chessboard_image;
  cv::undistort(chessboard_image, undistorted_chessboard_image, camera_matrix, distortion_coeffs);

  // test (0, 0, 0)
  cv::Mat origin_pixel = camera_matrix * translation_matrix;
  origin_pixel /= origin_pixel.at<double>(2, 0);
  cv::circle(undistorted_chessboard_image, cv::Point(origin_pixel.at<double>(0, 0), origin_pixel.at<double>(1, 0)), 5, cv::Scalar(0, 0, 255));

  // test (1, 0, 0)
  cv::Mat x_point = cv::Mat::zeros(3, 1, CV_64FC1);
  x_point.at<double>(0, 0) = 1;
  cv::Mat x_pixel = camera_matrix * (rotation_matrix * x_point + translation_matrix);
  x_pixel /= x_pixel.at<double>(2, 0);
  cv::circle(undistorted_chessboard_image, cv::Point(x_pixel.at<double>(0, 0), x_pixel.at<double>(1, 0)), 5, cv::Scalar(0, 255, 0));

  // test (0, 1, 0)
  cv::Mat y_point = cv::Mat::zeros(3, 1, CV_64FC1);
  y_point.at<double>(1, 0) = 1;
  cv::Mat y_pixel = camera_matrix * (rotation_matrix * y_point + translation_matrix);
  y_pixel /= y_pixel.at<double>(2, 0);
  cv::circle(undistorted_chessboard_image, cv::Point(y_pixel.at<double>(0, 0), y_pixel.at<double>(1, 0)), 5, cv::Scalar(255, 0, 0));

  // extract features(corners) from the undistorted images
  for (size_t i = 0; i < position_estimation_file_names.size(); ++i)
  {
    raw_image = cv::imread(position_estimation_directory + position_estimation_file_names[i]);
    cv::Mat undistorted_image;
    cv::undistort(raw_image, undistorted_image, camera_matrix, distortion_coeffs);

    cv::Mat undistorted_gray_image;
    cv::cvtColor(undistorted_image, undistorted_gray_image, cv::COLOR_BGR2GRAY);

    vector<cv::Point2f> feature_corners;
    cv::goodFeaturesToTrack(undistorted_gray_image, feature_corners, 3, 0.01, 10);

    // Estimate the 3d position of each feature in the world coordinates
    cv::Mat temp_image = undistorted_chessboard_image.clone();
    for (size_t j = 0; j < feature_corners.size(); ++j)
    {
      cout << "Feature " << j << ":" << endl;
      cout << "Image: " << endl << feature_corners[j].x << " " << feature_corners[j].y << endl;
      cv::circle(undistorted_image, cv::Point(feature_corners[j].x, feature_corners[j].y), 5, cv::Scalar(0, 0, 255));
      cv::circle(temp_image, cv::Point(feature_corners[j].x, feature_corners[j].y), 5, cv::Scalar(0, 0, 255));

      cv::Mat image_point(3, 1, CV_64FC1, 1.0);
      image_point.at<double>(0, 0) = feature_corners[j].x;
      image_point.at<double>(1, 0) = feature_corners[j].y;

      const cv::Mat temp = camera_matrix.inv() * image_point;
      const cv::Mat r_3 = (rotation_matrix.t())(cv::Rect(0, 2, 3, 1));
      const cv::Mat scale = (r_3 * translation_matrix) / (r_3 * temp);

      const cv::Mat world_point = rotation_matrix.t() * (scale.at<double>(0, 0) * temp - translation_matrix);
      cout << "World: " << endl << world_point << endl;
    }
    cv::imshow("Corner Detection", undistorted_image);
    cv::imshow("Undistorted Chessboard", temp_image);
    cv::waitKey(0);
  }

  return 0;
}
