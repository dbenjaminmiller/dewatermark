#include "dewatermark.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace fs = std::filesystem;

// Windows passes argv in the ANSI code page; paths are UTF-8 everywhere else in the program.
static std::vector<std::string> utf8Args(int argc, char** argv) {
#ifdef _WIN32
    int n = 0;
    LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(len > 0 ? len - 1 : 0, '\0');
        if (len > 1) WideCharToMultiByte(CP_UTF8, 0, w[i], -1, s.data(), len, nullptr, nullptr);
        out.push_back(s);
    }
    LocalFree(w);
    return out;
#else
    return {argv, argv + argc};
#endif
}

static const char* kHelp =
    "dewatermark version 1.0\n"
    "\n"
    "usage: dewatermark [options] [input directory]\n"
    "\n"
    "The program will automatically create a subdirectory \"dewatermark_out\" for output TIFFs.\n"
    "If an existing output subdirectory is present, a suffix (-1, -2, etc.) will be added.\n"
    "\n"
    "Options:\n"
    "  --out [directory]\t\tChoose custom output directory.\n"
    "  --save-watermark [x]\tSave watermark info as [x].yml and [x]_{1-4}.tif.\n"
    "  --load-watermark [x]\tLoad watermark from [x].yml and [x]_{1-4}.tif (skips estimation).\n"
    "  --max-est [n]\t\t\tMaximum number of pages [n] used to estimate the watermark.\n"
    "  --search [n]\t\t\tSearch for watermarks shifted by up to [n] pixels (default: 64).\n"
    "  --overwrite\t\t\tOverwrite existing outputs (if present).\n"
    "  --help\t\t\t\tShow this help.\n";

static void usage() {
    fputs(kHelp, stderr);
    exit(2);
}

static std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> v;
    for (auto& e : fs::directory_iterator(fs::u8path(dir))) {
        if (!e.is_regular_file()) continue;
        auto name = e.path().filename().u8string();
        if (name.empty() || name[0] == '.') continue;
        auto ext = e.path().extension().u8string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff" || ext == ".bmp")
            v.push_back(e.path().u8string());
    }
    std::sort(v.begin(), v.end());
    return v;
}

static fs::path outputDir(const fs::path& in, bool overwrite) {
    fs::path p = in / "dewatermark_out";
    for (int i = 1; !overwrite && fs::exists(p); ++i) p = in / ("dewatermark_out-" + std::to_string(i));
    return p;
}

int main(int argc, char** argv) {
    auto args = utf8Args(argc, argv);
    Options o;
    std::string in, out, saveW, loadW;
    bool overwrite = false;
    for (size_t i = 1; i < args.size(); ++i) {
        std::string a = args[i];
        auto next = [&]() -> std::string { if (i + 1 >= args.size()) usage(); return args[++i]; };
        if (a == "--help") { fputs(kHelp, stdout); return 0; }
        else if (a == "--out") out = next();
        else if (a == "--save-watermark") saveW = next();
        else if (a == "--load-watermark") loadW = next();
        else if (a == "--overwrite") overwrite = true;
        else if (a == "--search") o.search = std::max(8, atoi(next().c_str()));
        else if (a == "--max-est") o.maxEst = std::max(2, atoi(next().c_str()));
        else if (a.rfind("--", 0) == 0 || !in.empty()) usage();
        else in = a;
    }
    if (in.empty()) in = ".";
    auto files = listImages(in);
    if (files.empty()) { fprintf(stderr, "no images in %s\n", in.c_str()); return 1; }
    fs::path outDir = out.empty() ? outputDir(fs::u8path(in), overwrite) : fs::u8path(out);
    std::map<std::string, int> stems;
    for (auto& f : files) ++stems[fs::u8path(f).stem().u8string()];
    auto outputFor = [&](const std::string& f) {
        fs::path p = fs::u8path(f);
        return (outDir / (stems[p.stem().u8string()] > 1 ? p.filename() : p.stem())).u8string() + ".tif";
    };
    if (!overwrite)
        for (auto& f : files)
            if (fs::exists(fs::u8path(outputFor(f)))) {
                fprintf(stderr, "%s exists (use --overwrite to replace existing outputs)\n", outputFor(f).c_str());
                return 1;
            }

    Model m;
    std::vector<Xform> xf(files.size());
    std::vector<char> known(files.size(), 0);
    if (!loadW.empty()) {
        if (!loadModel(loadW, m)) { fprintf(stderr, "cannot load watermark %s\n", loadW.c_str()); return 1; }
    } else {
        std::vector<int> est;
        int n = (int)files.size(), ne = std::min(n, o.maxEst);
        for (int j = 0; j < ne; ++j) est.push_back((int)((long)j * n / ne));
        std::vector<std::string> ef;
        for (int k : est) ef.push_back(files[k]);
        std::vector<cv::Mat> grays(ne);
        cv::parallel_for_(cv::Range(0, ne), [&](const cv::Range& r) {
            for (int j = r.start; j < r.end; ++j) grays[j] = readImage(ef[j], cv::IMREAD_GRAYSCALE);
        });
        for (int j = 0; j < ne; ++j) if (grays[j].empty()) { fprintf(stderr, "cannot read %s\n", ef[j].c_str()); return 1; }
        fprintf(stderr, "registering %d pages\n", ne);
        std::vector<Xform> ex;
        registerSet(grays, m, ex, o);
        estimateWatermark(ef, grays, ex, m, o);
        for (int j = 0; j < ne; ++j) { xf[est[j]] = ex[j]; known[est[j]] = 1; }
    }
    if (!saveW.empty() && !saveModel(saveW, m)) { fprintf(stderr, "cannot save watermark %s\n", saveW.c_str()); return 1; }

    fs::create_directories(outDir);
    int nt = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<size_t> next{0};
    std::mutex logm;
    std::vector<std::string> absent;
    fprintf(stderr, "writing %zu pages to %s\n", files.size(), outDir.u8string().c_str());
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t) pool.emplace_back([&] {
        for (size_t k; (k = next++) < files.size();) {
            cv::Mat page = readImage(files[k], cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
            if (!page.empty() && page.depth() != CV_8U && page.depth() != CV_16U) page = readImage(files[k], cv::IMREAD_COLOR);
            if (page.empty()) { std::lock_guard<std::mutex> l(logm); fprintf(stderr, "cannot read %s\n", files[k].c_str()); continue; }
            Xform x = xf[k];
            if (!known[k]) {
                cv::Mat g;
                page.convertTo(g, CV_8U, page.depth() == CV_16U ? 1 / 257.0 : 1);
                cv::cvtColor(g, g, cv::COLOR_BGR2GRAY);
                x = registerPage(g, m, o);
            }
            cv::Mat J = removeWatermark(page, m, x), result;
            J.convertTo(result, page.type());
            auto dst = outputFor(files[k]);
            bool ok = writeImage(dst, result, {cv::IMWRITE_TIFF_COMPRESSION, 5 /* LZW */});
            std::lock_guard<std::mutex> l(logm);
            if (!ok) fprintf(stderr, "cannot write %s\n", dst.c_str());
            auto name = fs::u8path(files[k]).filename().u8string();
            if (!x.present) absent.push_back(name);
            fprintf(stderr, "  %-40s scale %.4f at %8.2f,%8.2f  ncc %.3f%s\n", name.c_str(), x.a, x.bx, x.by, x.ncc,
                    x.present ? "" : "  [no watermark found: copied unchanged]");
        }
    });
    for (auto& t : pool) t.join();
    fprintf(stderr, "done: %zu pages, %zu without watermark\n", files.size(), absent.size());
    return 0;
}
