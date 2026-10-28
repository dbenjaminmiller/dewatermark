#include "dewatermark.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

// Not cv::imread/imwrite: those take narrow paths, which break non-ASCII names on Windows.
cv::Mat readImage(const std::string& path, int flags) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    std::vector<uchar> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return buf.empty() ? cv::Mat() : cv::imdecode(buf, flags);
}

bool writeImage(const std::string& path, const cv::Mat& img, const std::vector<int>& params) {
    std::vector<uchar> buf;
    auto ext = std::filesystem::u8path(path).extension().u8string();
    if (!cv::imencode(ext, img, buf, params)) return false;
    std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
    return f.write(reinterpret_cast<const char*>(buf.data()), buf.size()).good();
}

bool saveModel(const std::string& prefix, const Model& m) {
    auto dir = std::filesystem::u8path(prefix).parent_path();
    if (!dir.empty()) std::filesystem::create_directories(dir);
    cv::FileStorage fs(prefix + ".yml", cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "refSize" << m.refSize << "align" << (int)m.align << "anchor" << m.anchor << "propScale" << (int)m.propScale
       << "origin" << m.origin << "frame" << m.frame << "fineRoi" << m.fineRoi << "medNcc" << m.medNcc
       << "inkClamp" << m.inkClamp;
    return writeImage(prefix + "_1.tif", m.gain) && writeImage(prefix + "_2.tif", m.offset) &&
           writeImage(prefix + "_3.tif", m.tmplFull) && writeImage(prefix + "_4.tif", m.tmplSmall);
}

bool loadModel(const std::string& prefix, Model& m) {
    cv::FileStorage fs(prefix + ".yml", cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    int prop = 0, align = 1;
    fs["refSize"] >> m.refSize; fs["align"] >> align; fs["anchor"] >> m.anchor; fs["propScale"] >> prop;
    fs["origin"] >> m.origin; fs["frame"] >> m.frame; fs["fineRoi"] >> m.fineRoi; fs["medNcc"] >> m.medNcc;
    fs["inkClamp"] >> m.inkClamp;
    m.propScale = prop;
    m.align = align;
    m.gain = readImage(prefix + "_1.tif", cv::IMREAD_UNCHANGED);
    m.offset = readImage(prefix + "_2.tif", cv::IMREAD_UNCHANGED);
    m.tmplFull = readImage(prefix + "_3.tif", cv::IMREAD_UNCHANGED);
    m.tmplSmall = readImage(prefix + "_4.tif", cv::IMREAD_UNCHANGED);
    return !m.gain.empty() && !m.offset.empty() && !m.tmplFull.empty() && !m.tmplSmall.empty();
}

static cv::Mat warpToPage(const cv::Mat& canon, const Xform& x, cv::Size page, double border) {
    cv::Mat M(cv::Matx23d(x.a, 0, x.bx, 0, x.a, x.by)), out;
    cv::warpAffine(canon, out, M, page, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(border));
    return out;
}

cv::Mat removeWatermark(const cv::Mat& page, const Model& m, const Xform& x) {
    cv::Mat I;
    page.convertTo(I, CV_32FC3);
    if (!x.present) return I;
    const float unit = page.depth() == CV_16U ? 257.f : 1.f;
    cv::Mat g = warpToPage(m.gain, x, page.size(), 1.0), t = warpToPage(m.offset, x, page.size(), 0.0) * unit;
    cv::max(g, 0.05, g);
    cv::parallel_for_(cv::Range(0, I.rows), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) {
            auto* p = I.ptr<cv::Vec3f>(y);
            auto* gg = g.ptr<cv::Vec3f>(y);
            auto* tt = t.ptr<cv::Vec3f>(y);
            for (int xx = 0; xx < I.cols; ++xx)
                for (int c = 0; c < 3; ++c)
                    if (gg[xx][c] != 1.f || tt[xx][c] != 0.f) p[xx][c] = (p[xx][c] - tt[xx][c]) / gg[xx][c];
        }
    });
    return I;
}

static cv::Mat touched(const cv::Mat& g, const cv::Mat& t) {
    cv::Mat mark(g.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < g.rows; ++y)
        for (int x = 0; x < g.cols; ++x)
            for (int c = 0; c < 3; ++c)
                if (std::fabs(g.at<cv::Vec3f>(y, x)[c] - 1) > 0.01f || std::fabs(t.at<cv::Vec3f>(y, x)[c]) > 0.5f)
                    mark.at<uchar>(y, x) = 255;
    return mark;
}

// A page carries the watermark if, where the watermark changes the paper's tone, the page looks more
// like the paper through the watermark than like bare paper. Ink looks like neither and has no say.
double watermarkFit(const cv::Mat& page, const Model& m, const Xform& x) {
    cv::Mat I;
    page.convertTo(I, CV_32FC3, page.depth() == CV_16U ? 1 / 257.0 : 1);
    cv::Mat g = warpToPage(m.gain, x, page.size(), 1.0), t = warpToPage(m.offset, x, page.size(), 0.0);
    cv::Mat mark = touched(g, t), near, ring;
    if (!cv::countNonZero(mark)) return 0;
    cv::dilate(mark, near, cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9}));
    cv::dilate(mark, ring, cv::getStructuringElement(cv::MORPH_ELLIPSE, {41, 41}));
    ring &= ~near;
    cv::Vec3f P;
    for (int c = 0; c < 3; ++c) {
        int hist[256] = {0}, total = 0;
        for (int y = 0; y < I.rows; ++y)
            for (int xx = 0; xx < I.cols; ++xx)
                if (ring.at<uchar>(y, xx)) { hist[std::clamp((int)I.at<cv::Vec3f>(y, xx)[c], 0, 255)]++; total++; }
        int v = 0;
        for (int acc = 0; v < 255 && (acc += hist[v]) < total / 2; ++v) {}
        P[c] = (float)v;
    }
    double yes = 0, all = 0;
    for (int y = 0; y < I.rows; ++y)
        for (int xx = 0; xx < I.cols; ++xx) {
            if (!mark.at<uchar>(y, xx)) continue;
            auto v = I.at<cv::Vec3f>(y, xx), gg = g.at<cv::Vec3f>(y, xx), tt = t.at<cv::Vec3f>(y, xx);
            double dp = 0, da = 0, w = 0;
            for (int c = 0; c < 3; ++c) {
                double with = gg[c] * P[c] + tt[c];
                dp += std::fabs(v[c] - with);
                da += std::fabs(v[c] - P[c]);
                w += std::fabs(with - P[c]);
            }
            if (w < 3 || std::min(dp, da) > w / 4) continue;
            all += w;
            if (dp < da) yes += w;
        }
    return all > 0 ? yes / all : 0;
}
