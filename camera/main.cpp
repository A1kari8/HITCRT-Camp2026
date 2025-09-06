#include <fstream>
#include <opencv2/opencv.hpp>

#include "json.hpp"

int main() {
    int boardWidth = 11;
    int boardHeight = 8;
    float squareSize = 1.f;
    cv::Size boardSize(boardWidth, boardHeight);

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point2f> corners;

    cv::VideoCapture cap("../../assets/calibration.mp4", cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件" << std::endl;
        return -1;
    }

    int frameCount = 0;
    cv::Mat frame, gray;
    cv::Size imageSize;

    std::vector<cv::Point2f> lastCorners;
    double minShift = 20.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        frameCount++;
        if (frameCount % 1 != 0) continue;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        bool found = cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
                cv::CALIB_CB_FAST_CHECK);

        if (found && corners.size() == boardWidth * boardHeight) {
            cv::Rect bounding = cv::boundingRect(corners);
            double area = bounding.width * bounding.height;
            if (area < 5000) continue;  // 面积太小

            if (!lastCorners.empty()) {
                double shift = 0;
                for (size_t i = 0; i < corners.size(); ++i) {
                    shift += cv::norm(corners[i] - lastCorners[i]);
                }
                shift /= corners.size();
                if (shift < minShift) continue;
            }

            lastCorners = corners;

            if (imageSize.width == 0 || imageSize.height == 0) {
                imageSize = frame.size();  // 只在第一次成功时设置
            }

            cv::cornerSubPix(
                gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(
                    cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            std::vector<cv::Point3f> objectCorners;
            for (int j = 0; j < boardHeight; j++) {
                for (int k = 0; k < boardWidth; k++) {
                    objectCorners.emplace_back(k * squareSize, j * squareSize,
                                               0);
                }
            }

            objectPoints.push_back(objectCorners);
            imagePoints.push_back(corners);

            cv::drawChessboardCorners(frame, boardSize, corners, found);
            cv::imshow("image", frame);
            cv::waitKey(100);

            std::cout << "保留高质量帧 #" << frameCount
                      << "，角点面积: " << area << std::endl;
        }
    }

    if (imagePoints.empty()) {
        std::cerr << "没有检测棋盘格" << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    std::cout << "标定前图像尺寸: " << imageSize.width << " x "
              << imageSize.height << std::endl;

    if (imageSize.width <= 0 || imageSize.height <= 0) {
        std::cerr << "图像尺寸无效" << std::endl;
        return -1;
    }

    cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                        distCoeffs, rvecs, tvecs);

    double totalErr = 0;
    int totalPoints = 0;
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix,
                          distCoeffs, projectedPoints);
        double err = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
        totalErr += err * err;
        totalPoints += objectPoints[i].size();
    }

    double meanErr = std::sqrt(totalErr / totalPoints);
    std::cout << "重投影误差: " << meanErr << std::endl;

    nlohmann::json j;
    j["camera_matrix"] = {
        {cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(0, 1),
         cameraMatrix.at<double>(0, 2)},
        {cameraMatrix.at<double>(1, 0), cameraMatrix.at<double>(1, 1),
         cameraMatrix.at<double>(1, 2)},
        {cameraMatrix.at<double>(2, 0), cameraMatrix.at<double>(2, 1),
         cameraMatrix.at<double>(2, 2)}};
    j["dist_coeffs"] = {distCoeffs.at<double>(0), distCoeffs.at<double>(1),
                        distCoeffs.at<double>(2), distCoeffs.at<double>(3),
                        distCoeffs.at<double>(4)};

    std::ofstream out("../../assets/calibration.json");
    out << j.dump(4);

    std::cout << "Camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients:\n" << distCoeffs << std::endl;

    return 0;
}
