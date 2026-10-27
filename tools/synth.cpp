#include "dewatermark.hpp"

#include <cstdio>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

static cv::Mat place(const cv::Mat& canon, const Xform& x, cv::Size page, double border) {
    cv::Mat M(cv::Matx23d(x.a, 0, x.bx, 0, x.a, x.by)), out;
    cv::warpAffine(canon, out, M, page, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(border));
    return out;
}

static cv::Mat apply(const cv::Mat& clean, const Model& m, const Xform& x) {
    cv::Mat J, I, out;
    clean.convertTo(J, CV_32FC3);
    I = J.mul(place(m.gain, x, clean.size(), 1)) + place(m.offset, x, clean.size(), 0);
    I.convertTo(out, CV_8UC3);
    return out;
}

static cv::Mat resized(const cv::Mat& img, double f) {
    cv::Mat out;
    cv::resize(img, out, cv::Size((int)std::lround(img.cols * f), (int)std::lround(img.rows * f)), 0, 0,
               f < 1 ? cv::INTER_AREA : cv::INTER_CUBIC);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: synth shift|scale <clean_dir> <watermark_prefix> <out_dir> <truth_dir> [count] [absent_index] [seed]\n");
        return 2;
    }
    std::string mode = argv[1];
    int count = argc > 6 ? atoi(argv[6]) : 1 << 30, absent = argc > 7 ? atoi(argv[7]) : -1;
    std::mt19937 rng(argc > 8 ? atoi(argv[8]) : 1);
    Model m;
    if (!loadModel(argv[3], m)) { fprintf(stderr, "cannot load watermark %s\n", argv[3]); return 1; }
    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(argv[2])) if (e.path().extension() == ".tif") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    fs::create_directories(argv[4]);
    fs::create_directories(argv[5]);
    std::uniform_real_distribution<double> res(0.85, 1.15), zoom(0.8, 1.25);
    std::uniform_int_distribution<int> jitter(-30, 30);
    for (int k = 0; k < (int)files.size() && k < count; ++k) {
        cv::Mat clean = readImage(files[k], cv::IMREAD_COLOR), truth, marked;
        if (mode == "shift") {
            truth = resized(clean, res(rng));
            Xform x;
            x.bx = std::round((truth.cols - m.frame.width) / 2.0) + jitter(rng);
            x.by = std::round((truth.rows - m.frame.height) / 2.0) + jitter(rng);
            marked = k == absent ? truth : apply(truth, m, x);
            fprintf(stderr, "  placed at %.0f,%.0f\n", x.bx, x.by);
        } else {
            double f = zoom(rng);
            truth = resized(clean, f);
            marked = k == absent ? truth : resized(apply(clean, m, initialXform(m, clean.size())), f);
        }
        auto stem = fs::path(files[k]).stem().string();
        writeImage((fs::path(argv[4]) / (stem + ".jpg")).string(), marked, {cv::IMWRITE_JPEG_QUALITY, 92});
        writeImage((fs::path(argv[5]) / (stem + ".png")).string(), truth);
        fprintf(stderr, "%s %dx%d%s\n", stem.c_str(), truth.cols, truth.rows, k == absent ? " (no watermark)" : "");
    }
    return 0;
}
