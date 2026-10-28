// Per pixel and channel: page = gain * original + offset, so original = (page - offset) / gain.
#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// page = a * canonical + b
struct Xform {
    double a = 1, bx = 0, by = 0;
    double ncc = 0;
    bool present = true;
};

enum Anchor { TL, TR, BL, BR, CENTER, NUM_ANCHORS };
const char* anchorName(int anchor);

struct Options {
    int coarse = 4;
    int search = 64;
    double scaleRange = 0.03;
    double hpSigma = 8;
    double fineSigma = 3;
    double presentRatio = 0.4;
    int fineRoi = 4096;
    int maxEst = 80;

    int strokeMax = 41;
    double detectThresh = 6;
    double bgSigma = 20;
    int passes = 3;
    double alphaMin = 0.005;

    int inkSize = 31;
    double inkLevel = 24;
    int block = 16;
    double flatMaxGain = 0.9;
};

struct Model {
    cv::Size refSize;
    bool align = true;
    int anchor = TL;
    bool propScale = false;
    cv::Point origin;
    cv::Size frame;
    cv::Mat tmplSmall, tmplFull;
    cv::Rect fineRoi;
    double medNcc = 0;
    double inkClamp = 255;
    cv::Mat gain, offset;
};

inline bool sampleAt(const cv::Mat& g, double sx, double sy, float& v) {
    if (sx < 0 || sy < 0 || sx > g.cols - 1 || sy > g.rows - 1) return false;
    int x0 = (int)sx, y0 = (int)sy;
    double fx = sx - x0, fy = sy - y0;
    const uchar* r0 = g.ptr<uchar>(y0);
    if (fx == 0 && fy == 0) { v = r0[x0]; return true; }
    int x1 = std::min(x0 + 1, g.cols - 1), y1 = std::min(y0 + 1, g.rows - 1);
    const uchar* r1 = g.ptr<uchar>(y1);
    v = float((1 - fy) * ((1 - fx) * r0[x0] + fx * r0[x1]) + fy * ((1 - fx) * r1[x0] + fx * r1[x1]));
    return true;
}

inline cv::Rect maskBounds(const cv::Mat& m) {
    cv::Mat nz;
    cv::findNonZero(m, nz);
    if (nz.empty()) return {};
    int x0 = m.cols, y0 = m.rows, x1 = -1, y1 = -1;
    for (int i = 0; i < (int)nz.total(); ++i) {
        cv::Point p = nz.at<cv::Point>(i);
        x0 = std::min(x0, p.x); x1 = std::max(x1, p.x); y0 = std::min(y0, p.y); y1 = std::max(y1, p.y);
    }
    return {x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}

cv::Mat readImage(const std::string& path, int flags);
bool writeImage(const std::string& path, const cv::Mat& img, const std::vector<int>& params = {});

cv::Mat highpass(const cv::Mat& src, double sigma);
Xform initialXform(const Model& m, cv::Size page);
double paperLevel(const cv::Mat& channel);
void stackMedian(const std::vector<cv::Mat>& g, const std::vector<Xform>& xf, double div, cv::Size frame,
                 cv::Mat& med, cv::Mat& count, cv::Mat* darkest = nullptr, cv::Mat* spread = nullptr, bool skipZero = false);
cv::Mat watermarkAgreement(const cv::Mat& med, const cv::Mat& spread, const cv::Mat& valid, int wide, float& paper,
                           double tolerance = 0.25);
void registerSet(const std::vector<cv::Mat>& grays, Model& m, std::vector<Xform>& xf, const Options& o);
Xform registerPage(const cv::Mat& gray, const Model& m, const Options& o);
struct WatermarkLine {
    cv::Mat tmpl;
    cv::Rect box;
};
WatermarkLine watermarkLine(const Model& m);
Xform refineToLine(const cv::Mat& gray, const Model& m, const WatermarkLine& L, Xform x);

void estimateWatermark(const std::vector<std::string>& files, const std::vector<cv::Mat>& grays,
                       const std::vector<Xform>& xf, Model& m, const Options& o);

bool saveModel(const std::string& prefix, const Model& m);
bool loadModel(const std::string& prefix, Model& m);
cv::Mat removeWatermark(const cv::Mat& page, const Model& m, const Xform& x);
double watermarkFit(const cv::Mat& page, const Model& m, const Xform& x);
