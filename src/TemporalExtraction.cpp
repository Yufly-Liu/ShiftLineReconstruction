#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;

// ======== 参数配置 ========
const int Movenum = 40;        // 条纹数量
const int Gray_order = 6;      // Gray码阶数
const double gaussian_sigma = 1.0;
const int gaussian_size = 10;

// ======== 工具函数 ========

// 生成高斯滤波器
Mat gaussianFilter(int ksize, double sigma) {
    return getGaussianKernel(ksize, sigma, CV_64F) * getGaussianKernel(ksize, sigma, CV_64F).t();
}

// 灰度重心法求中心
double grayCentroid(const vector<double>& win, int pos_j, int half_width, int Movenum) {
    double sumI = 0.0, sumIj = 0.0;
    for (int kx = pos_j - half_width; kx <= pos_j + half_width; kx++) {
        if (kx >= 0 && kx < (int)win.size()) {
            sumI += win[kx];
            sumIj += win[kx] * kx;
        }
    }
    if (sumI == 0) return 0;
    double new_j = sumIj / sumI - Movenum / 2.0;
    if (new_j < 0) new_j += Movenum;
    if (new_j > Movenum) new_j -= Movenum;
    return new_j;
}

// ======== 主函数 ========
int main() {
    string prefix = "";
    vector<Mat> line_list;
    Mat h = gaussianFilter(gaussian_size, gaussian_sigma);

    // 读取40张图像并滤波
    for (int i = 0; i < Movenum; i++) {
        string name = prefix + to_string(i) + ".bmp";
        Mat img = imread(name, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "无法读取图片: " << name << endl;
            return -1;
        }
        img.convertTo(img, CV_64F);
        Mat filtered;
        filter2D(img, filtered, -1, h);
        line_list.push_back(filtered);
    }

    int km = line_list[0].rows;
    int kn = line_list[0].cols;

    // 均值图像
    Mat I_temp = Mat::zeros(km, kn, CV_64F);
    for (int k = 0; k < Movenum; k++)
        I_temp += line_list[k] / Movenum;

    Mat PP = Mat::zeros(km, kn, CV_64F);
    for (int i = 0; i < km; i++)
        for (int j = 0; j < kn; j++)
            if (I_temp.at<double>(i, j) < 200 && I_temp.at<double>(i, j) > 1)
                PP.at<double>(i, j) = 1;

    // ====== 时序中心提取 ======
    Mat temporal_center = Mat::zeros(km, kn, CV_64F);
    Mat P2 = Mat::zeros(km, kn, CV_64F);

    for (int i = 0; i < km; i++) {
        for (int j = 0; j < kn; j++) {
            if (PP.at<double>(i, j) <= 0) continue;

            vector<double> points_40(Movenum);
            for (int k = 0; k < Movenum; k++)
                points_40[k] = line_list[k].at<double>(i, j);

            // 检查是否只有一个峰值
            int count255 = 0;
            for (double v : points_40) if (v >= 255.0) count255++;

            vector<double> pixel_win(points_40.begin() + Movenum / 2, points_40.end());
            pixel_win.insert(pixel_win.end(), points_40.begin(), points_40.end());
            pixel_win.insert(pixel_win.end(), points_40.begin(), points_40.begin() + Movenum / 2);

            if (count255 < 2) {
                // 单峰灰度重心法
                auto max_it = max_element(points_40.begin(), points_40.end());
                int index_max = (int)(max_it - points_40.begin());
                int pos_j = index_max + Movenum / 2;
                double new_j = grayCentroid(pixel_win, pos_j, 5, Movenum);
                temporal_center.at<double>(i, j) = new_j;
                P2.at<double>(i, j) = 1;
            }
        }
    }

    cout << "中心提取完成" << endl;

    // ====== Gray码部分 ======
    vector<Mat> Gray_list;
    for (int i = Movenum; i < Movenum + Gray_order; i++) {
        string name = prefix + to_string(i) + ".bmp";
        Mat img = imread(name, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "无法读取Gray图: " << name << endl;
            return -1;
        }
        img.convertTo(img, CV_64F);
        Gray_list.push_back(img);
    }

    Mat I_b = imread(prefix + to_string(Movenum + Gray_order) + ".bmp", IMREAD_GRAYSCALE);
    Mat I_w = imread(prefix + to_string(Movenum + Gray_order + 1) + ".bmp", IMREAD_GRAYSCALE);
    I_b.convertTo(I_b, CV_64F);
    I_w.convertTo(I_w, CV_64F);

    Mat Thresh = (I_b + I_w) * 0.5;

    for (auto& g : Gray_list)
        g = (g - Thresh) > 0;

    // Gray码解码
    for (int i = Gray_order - 2; i >= 0; i--)
        Gray_list[i] = abs(Gray_list[i + 1] - Gray_list[i]);

    Mat Code_value1 = Mat::zeros(km, kn, CV_64F);
    Mat Code_value2 = Mat::zeros(km, kn, CV_64F);
    for (int i = 0; i < Gray_order; i++) {
        Code_value1 += Gray_list[i].mul(pow(2.0, i));
        if (i > 0) Code_value2 += Gray_list[i].mul(pow(2.0, i - 1));
    }

    Mat wphase = temporal_center - Movenum / 2.0;
    Mat unwph = Mat::zeros(km, kn, CV_64F);

    for (int i = 0; i < km; i++) {
        for (int j = 0; j < kn; j++) {
            double wp = wphase.at<double>(i, j);
            double N1 = Code_value1.at<double>(i, j);
            double N2 = Code_value2.at<double>(i, j);
            double val = N1 - N2;
            if (wp > Movenum / 4.0)
                unwph.at<double>(i, j) = wp + val * Movenum;
            else if (wp < -Movenum / 4.0)
                unwph.at<double>(i, j) = wp + (val + 1) * Movenum;
            else
                unwph.at<double>(i, j) = wp + N2 * Movenum;
        }
    }

    // 保存结果
    FileStorage fs("result.yml", FileStorage::WRITE);
    fs << "un_temporal_center" << unwph;
    fs.release();

    cout << "解码完成并保存 result.yml" << endl;
    return 0;
}
