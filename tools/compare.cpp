#include <opencv2/opencv.hpp>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

static double psnr(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask) {
    cv::Mat d;
    cv::absdiff(a, b, d);
    d.convertTo(d, CV_32F);
    cv::Scalar m = mask.empty() ? cv::mean(d.mul(d)) : cv::mean(d.mul(d), mask);
    double mse = (m[0] + m[1] + m[2]) / 3;
    return mse <= 0 ? 99 : 10 * std::log10(255.0 * 255.0 / mse);
}

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: compare <input_dir> <output_dir> <truth_dir>\n"); return 2; }
    std::vector<fs::path> truths;
    for (auto& e : fs::directory_iterator(argv[3])) if (e.path().extension() == ".png") truths.push_back(e.path());
    std::sort(truths.begin(), truths.end());
    double si = 0, so = 0, wi = 0, wo = 0;
    int n = 0, nw = 0;
    printf("%-12s %12s %12s %12s %12s\n", "page", "in(all)", "out(all)", "in(wm)", "out(wm)");
    for (auto& t : truths) {
        auto stem = t.stem().string();
        cv::Mat T = cv::imread(t.string()), I = cv::imread((fs::path(argv[1]) / (stem + ".jpg")).string()),
                O = cv::imread((fs::path(argv[2]) / (stem + ".tif")).string());
        if (T.empty() || I.empty() || O.empty() || T.size() != O.size()) { printf("%-12s missing\n", stem.c_str()); continue; }
        cv::Mat d, g;
        cv::absdiff(I, T, d);
        cv::cvtColor(d, g, cv::COLOR_BGR2GRAY);
        cv::Mat wm = g > 12;
        cv::dilate(wm, wm, cv::Mat::ones(5, 5, CV_8U));
        double a = psnr(I, T, {}), b = psnr(O, T, {});
        si += a; so += b; ++n;
        if (cv::countNonZero(wm) > 1000) {
            double c = psnr(I, T, wm), e = psnr(O, T, wm);
            wi += c; wo += e; ++nw;
            printf("%-12s %12.2f %12.2f %12.2f %12.2f\n", stem.c_str(), a, b, c, e);
        } else {
            printf("%-12s %12.2f %12.2f %12s %12s\n", stem.c_str(), a, b, "-", "-");
        }
    }
    if (n) printf("%-12s %12.2f %12.2f %12.2f %12.2f\n", "mean", si / n, so / n, nw ? wi / nw : 0, nw ? wo / nw : 0);
    return 0;
}
