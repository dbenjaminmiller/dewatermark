#include "dewatermark.hpp"

#include <algorithm>
#include <functional>
#include <cstdio>

static float quantile(std::vector<float>& v, double q) {
    size_t i = std::min(v.size() - 1, (size_t)(q * v.size()));
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
}

static cv::Mat ellipse(int d) { return cv::getStructuringElement(cv::MORPH_ELLIPSE, {d, d}); }

static void warpToCanon(const cv::Mat& page, const Xform& x, cv::Rect roi, cv::Mat& out, cv::Mat& valid) {
    cv::Mat M(cv::Matx23d(x.a, 0, x.bx + x.a * roi.x, 0, x.a, x.by + x.a * roi.y));
    cv::warpAffine(page, out, M, roi.size(), cv::INTER_CUBIC | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);
    cv::Mat ones(page.size(), CV_8U, cv::Scalar(255));
    cv::warpAffine(ones, valid, M, roi.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);
}

static cv::Mat inkContrast(const cv::Mat& img, int size) {
    cv::Mat closed;
    cv::morphologyEx(img, closed, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {size, size}));
    return closed - img;
}

static cv::Mat fillHoles(const cv::Mat& m) {
    cv::Mat pad;
    cv::copyMakeBorder(~m, pad, 1, 1, 1, 1, cv::BORDER_CONSTANT, 255);
    cv::floodFill(pad, cv::Point(0, 0), 0);
    return m | pad(cv::Rect(1, 1, m.cols, m.rows));
}

static cv::Mat componentsTouching(const cv::Mat& mask, const cv::Mat& seed) {
    cv::Mat lab;
    int nl = cv::connectedComponents(mask, lab, 8);
    std::vector<char> hit(nl, 0);
    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            if (seed.at<uchar>(y, x)) hit[lab.at<int>(y, x)] = 1;
    hit[0] = 0;
    cv::Mat out(mask.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            if (hit[lab.at<int>(y, x)]) out.at<uchar>(y, x) = 255;
    return out;
}

static cv::Mat flatRegion(const cv::Mat& seed, const cv::Mat& ink, const cv::Mat& valid, const cv::Mat& thin,
                          const Options& o, double& contrast) {
    cv::Mat c, excl;
    cv::dilate(ink, c, ellipse(9));
    cv::dilate(thin, excl, ellipse(9));
    excl |= ~valid;

    const int B = o.block, gw = (ink.cols + B - 1) / B, gh = (ink.rows + B - 1) / B;
    cv::Mat q(gh, gw, CV_32F, cv::Scalar(0)), dense(gh, gw, CV_8U, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, gh), [&](const cv::Range& r) {
        std::vector<float> v;
        for (int by = r.start; by < r.end; ++by)
            for (int bx = 0; bx < gw; ++bx) {
                v.clear();
                for (int y = by * B; y < std::min(ink.rows, (by + 1) * B); ++y)
                    for (int x = bx * B; x < std::min(ink.cols, (bx + 1) * B); ++x)
                        if (!excl.at<uchar>(y, x)) v.push_back(c.at<float>(y, x));
                if ((int)v.size() < B * B / 4) continue;
                float top = quantile(v, 0.95);
                q.at<float>(by, bx) = top;
                if (quantile(v, 0.5) >= 0.75f * top) dense.at<uchar>(by, bx) = 255;
            }
    });
    std::vector<float> all;
    for (int i = 0; i < (int)q.total(); ++i) if (dense.at<uchar>(i)) all.push_back(q.at<float>(i));
    contrast = 0;
    if (!all.empty()) {
        float top = quantile(all, 0.99);
        std::vector<float> inky;
        for (float v : all) if (v > 0.3f * top) inky.push_back(v);
        contrast = quantile(inky, 0.5);
    }
    cv::Mat R = seed.clone();
    if (contrast > 0) {
        cv::Mat s = q / contrast, weak = dense & (s > 0.3) & (s < o.flatMaxGain), seedGrid, weakFull;
        cv::resize(seed, seedGrid, cv::Size(gw, gh), 0, 0, cv::INTER_AREA);
        cv::dilate(seedGrid > 0, seedGrid, ellipse(9));
        cv::resize(componentsTouching(weak, seedGrid), weakFull, cv::Size(gw * B, gh * B), 0, 0, cv::INTER_NEAREST);
        R |= weakFull(cv::Rect(0, 0, ink.cols, ink.rows));
    }
    cv::morphologyEx(R, R, cv::MORPH_CLOSE, ellipse(4 * B + 1));
    return fillHoles(R) & valid;
}

static void estimateFlat(const std::vector<std::string>& files, const std::vector<Xform>& xf, const cv::Mat& R,
                         double grayContrast, int inkSize, Model& m) {
    const int ring = 96;
    cv::Rect inner = maskBounds(R), frame(0, 0, m.frame.width, m.frame.height);
    cv::Rect bb = cv::Rect(inner.x - ring, inner.y - ring, inner.width + 2 * ring, inner.height + 2 * ring) & frame;
    std::vector<cv::Mat> crop, ink, valid;
    for (size_t k = 0; k < files.size(); ++k) {
        if (!xf[k].present) continue;
        cv::Mat c, v;
        warpToCanon(readImage(files[k], cv::IMREAD_COLOR), xf[k], bb, c, v);
        crop.push_back(c);
        ink.push_back(inkContrast(c, inkSize));
        valid.push_back(v);
    }
    cv::Mat hi(bb.size(), CV_32FC3, cv::Scalar::all(0)), top(bb.size(), CV_32FC3, cv::Scalar::all(0));
    cv::Mat cnt(bb.size(), CV_32S, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, bb.height), [&](const cv::Range& r) {
        std::vector<float> v;
        for (int y = r.start; y < r.end; ++y)
            for (int x = 0; x < bb.width; ++x)
                for (int j = 0; j < 3; ++j) {
                    v.clear();
                    float t = 0;
                    for (size_t k = 0; k < crop.size(); ++k)
                        if (valid[k].at<uchar>(y, x)) {
                            v.push_back(crop[k].at<cv::Vec3b>(y, x)[j]);
                            t = std::max(t, (float)ink[k].at<cv::Vec3b>(y, x)[j]);
                        }
                    if (v.empty()) continue;
                    hi.at<cv::Vec3f>(y, x)[j] = quantile(v, 0.5);
                    top.at<cv::Vec3f>(y, x)[j] = t;
                    cnt.at<int>(y, x) = (int)v.size();
                }
    });
    crop.clear();
    ink.clear();
    valid.clear();

    cv::Mat Rc = R(bb), topPool, ringMask = ~Rc & (cnt >= 2);
    cv::dilate(top, topPool, ellipse(9));
    cv::Vec3d paper, contrast;
    for (int j = 0; j < 3; ++j) {
        std::vector<float> h, c;
        for (int y = 0; y < bb.height; ++y)
            for (int x = 0; x < bb.width; ++x)
                if (ringMask.at<uchar>(y, x)) {
                    h.push_back(hi.at<cv::Vec3f>(y, x)[j]);
                    c.push_back(topPool.at<cv::Vec3f>(y, x)[j]);
                }
        paper[j] = h.empty() ? 255 : quantile(h, 0.5);
        contrast[j] = c.size() < 1000 ? grayContrast : std::max(20.f, quantile(c, 0.95));
    }
    fprintf(stderr, "  flat layer %dx%d: paper %.0f/%.0f/%.0f, ink contrast %.0f/%.0f/%.0f\n", inner.width, inner.height,
            paper[0], paper[1], paper[2], contrast[0], contrast[1], contrast[2]);

    cv::Mat hiG;
    cv::transform(hi, hiG, cv::Matx13f(1 / 3.f, 1 / 3.f, 1 / 3.f));
    cv::Mat g(bb.size(), CV_32FC3, cv::Scalar::all(0)), conf(bb.size(), CV_32F, cv::Scalar(0));
    const int rad = 4;
    cv::parallel_for_(cv::Range(0, bb.height), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y)
            for (int x = 0; x < bb.width; ++x) {
                if (!Rc.at<uchar>(y, x)) continue;
                float h = hiG.at<float>(y, x);
                cv::Vec3f strongest(0, 0, 0);
                for (int yy = std::max(0, y - rad); yy <= std::min(bb.height - 1, y + rad); ++yy)
                    for (int xx = std::max(0, x - rad); xx <= std::min(bb.width - 1, x + rad); ++xx) {
                        if (!Rc.at<uchar>(yy, xx) || std::fabs(hiG.at<float>(yy, xx) - h) > 3) continue;
                        auto t = top.at<cv::Vec3f>(yy, xx);
                        for (int j = 0; j < 3; ++j) strongest[j] = std::max(strongest[j], t[j]);
                    }
                cv::Vec3f s;
                for (int j = 0; j < 3; ++j) s[j] = float(strongest[j] / contrast[j]);
                g.at<cv::Vec3f>(y, x) = s;
                if ((s[0] + s[1] + s[2]) / 3 >= 0.3f) conf.at<float>(y, x) = 1;
            }
    });
    {
        const int rr = 12;
        cv::Mat sm = g.clone();
        cv::parallel_for_(cv::Range(0, bb.height), [&](const cv::Range& r) {
            std::vector<float> v[3];
            for (int y = r.start; y < r.end; ++y)
                for (int x = 0; x < bb.width; ++x) {
                    if (!conf.at<float>(y, x)) continue;
                    float h = hiG.at<float>(y, x);
                    for (auto& e : v) e.clear();
                    for (int yy = std::max(0, y - rr); yy <= std::min(bb.height - 1, y + rr); yy += 2)
                        for (int xx = std::max(0, x - rr); xx <= std::min(bb.width - 1, x + rr); xx += 2)
                            if (conf.at<float>(yy, xx) && std::fabs(hiG.at<float>(yy, xx) - h) <= 3)
                                for (int j = 0; j < 3; ++j) v[j].push_back(g.at<cv::Vec3f>(yy, xx)[j]);
                    if (v[0].empty()) continue;
                    for (int j = 0; j < 3; ++j) sm.at<cv::Vec3f>(y, x)[j] = quantile(v[j], 0.5);
                }
        });
        g = sm;
    }
    cv::Mat c3;
    cv::merge(std::vector<cv::Mat>{conf, conf, conf}, c3);
    cv::Mat measured = g.mul(c3), filled = measured.clone(), have = conf.clone();
    for (double sigma : {6.0, 30.0, 120.0}) {
        cv::Mat num, den;
        cv::GaussianBlur(measured, num, {0, 0}, sigma);
        cv::GaussianBlur(conf, den, {0, 0}, sigma);
        for (int y = 0; y < bb.height; ++y)
            for (int x = 0; x < bb.width; ++x)
                if (!have.at<float>(y, x) && den.at<float>(y, x) > 0.02f) {
                    filled.at<cv::Vec3f>(y, x) = num.at<cv::Vec3f>(y, x) / den.at<float>(y, x);
                    have.at<float>(y, x) = 1;
                }
    }
    cv::Mat G = m.gain(bb), O = m.offset(bb);
    int nconf = 0, nid = 0, n = cv::countNonZero(Rc);
    for (int y = 0; y < bb.height; ++y)
        for (int x = 0; x < bb.width; ++x) {
            if (!Rc.at<uchar>(y, x)) continue;
            nconf += conf.at<float>(y, x) > 0;
            auto f = filled.at<cv::Vec3f>(y, x);
            auto h = hi.at<cv::Vec3f>(y, x);
            cv::Vec3f gg, t;
            float gm = 0, dm = 0;
            for (int j = 0; j < 3; ++j) {
                gg[j] = have.at<float>(y, x) ? std::clamp(f[j], 0.05f, 1.f) : 1.f;
                t[j] = float(h[j] - gg[j] * paper[j]);
                gm += gg[j] / 3;
                dm += float(std::fabs(h[j] - paper[j]) / 3);
            }
            if (gm > 0.97f && dm < 6) { ++nid; continue; }
            G.at<cv::Vec3f>(y, x) = gg;
            O.at<cv::Vec3f>(y, x) = t;
        }
    fprintf(stderr, "  flat layer: %d px, %.0f%% measured through ink, %d px indistinguishable from no watermark\n",
            n, 100.0 * nconf / std::max(1, n), nid);
}

static void estimateThin(const std::vector<std::string>& files, const std::vector<Xform>& xf, const cv::Mat& M,
                         const cv::Mat& excl, const cv::Mat& shared, double initialColor, Model& m, const Options& o) {
    cv::Rect wr = maskBounds(M);
    int pad = (int)std::ceil(3 * o.bgSigma);
    wr = cv::Rect(wr.x - pad, wr.y - pad, wr.width + 2 * pad, wr.height + 2 * pad) &
         cv::Rect(0, 0, m.frame.width, m.frame.height);
    cv::Mat Mw = M(wr), keep8 = ~excl(wr), keepf;
    keep8.convertTo(keepf, CV_32F, 1 / 255.0);
    double W[3] = {initialColor, initialColor, initialColor};
    cv::Mat alpha(wr.size(), CV_32FC3, cv::Scalar::all(0));
    for (int pass = 0; pass < o.passes; ++pass) {
        cv::Mat Sn = cv::Mat::zeros(wr.size(), CV_64FC3), Sd = Sn.clone(), Wn = Sn.clone(), Wd = Sn.clone();
        for (size_t k = 0; k < files.size(); ++k) {
            if (!xf[k].present) continue;
            cv::Mat page = readImage(files[k], cv::IMREAD_COLOR), gray, raw, v, I, vf, kv, ink, inkV, clear;
            cv::cvtColor(page, gray, cv::COLOR_BGR2GRAY);
            warpToCanon(page, xf[k], wr, raw, v);
            warpToCanon(inkContrast(gray, o.inkSize), xf[k], wr, ink, inkV);
            raw.convertTo(I, CV_32FC3);
            v.convertTo(vf, CV_32F, 1 / 255.0);
            // The page's own ink, beyond what every page shares, is neither paper to refer to nor a
            // view of the watermark.
            cv::Mat content = ink - shared(wr) > o.inkLevel;
            cv::Mat(~content).convertTo(clear, CV_32F, 1 / 255.0);
            cv::Mat k1 = keepf.mul(vf).mul(clear);
            cv::merge(std::vector<cv::Mat>{k1, k1, k1}, kv);
            cv::Mat Ik = I.mul(kv), num, den, num2;
            cv::GaussianBlur(Ik, num, {0, 0}, o.bgSigma);
            cv::GaussianBlur(kv, den, {0, 0}, o.bgSigma);
            cv::GaussianBlur(Ik.mul(I), num2, {0, 0}, o.bgSigma);
            cv::Mat R = num / (den + 1e-6), var = num2 / (den + 1e-6) - R.mul(R);
            cv::parallel_for_(cv::Range(0, wr.height), [&](const cv::Range& r) {
                for (int y = r.start; y < r.end; ++y)
                    for (int x = 0; x < wr.width; ++x) {
                        if (!Mw.at<uchar>(y, x) || !v.at<uchar>(y, x) || content.at<uchar>(y, x)) continue;
                        float d = den.at<cv::Vec3f>(y, x)[0];
                        if (d < 0.05f) continue;
                        auto i = I.at<cv::Vec3f>(y, x), rr = R.at<cv::Vec3f>(y, x), vv = var.at<cv::Vec3f>(y, x);
                        auto a = alpha.at<cv::Vec3f>(y, x);
                        auto& sn = Sn.at<cv::Vec3d>(y, x); auto& sd = Sd.at<cv::Vec3d>(y, x);
                        auto& wn = Wn.at<cv::Vec3d>(y, x); auto& wd = Wd.at<cv::Vec3d>(y, x);
                        for (int c = 0; c < 3; ++c) {
                            double g = W[c] - rr[c], e = i[c] - rr[c];
                            double w = d / (std::max(vv[c], 0.f) + 4.0);
                            if (pass > 0) {
                                double res = (e - a[c] * g) / 12.0;
                                w = std::fabs(res) < 1 ? w * (1 - res * res) * (1 - res * res) : 0;
                            }
                            sn[c] += w * e * g; sd[c] += w * g * g;
                            if (a[c] > 0.3f) { wn[c] += w * a[c] * (e + a[c] * rr[c]); wd[c] += w * a[c] * a[c]; }
                        }
                    }
            });
        }
        for (int y = 0; y < wr.height; ++y)
            for (int x = 0; x < wr.width; ++x) {
                if (!Mw.at<uchar>(y, x)) continue;
                for (int c = 0; c < 3; ++c) {
                    double d = Sd.at<cv::Vec3d>(y, x)[c];
                    alpha.at<cv::Vec3f>(y, x)[c] = d > 0 ? (float)std::clamp(Sn.at<cv::Vec3d>(y, x)[c] / d, 0.0, 0.95) : 0.f;
                }
            }
        if (pass == 1 && o.passes > 2) {
            cv::Scalar n = cv::sum(Wn), d = cv::sum(Wd);
            for (int c = 0; c < 3; ++c) if (d[c] > 0) W[c] = std::clamp(n[c] / d[c], 0.0, 255.0);
        }
        fprintf(stderr, "  thin strokes pass %d/%d\n", pass + 1, o.passes);
    }
    {
        cv::Mat smooth;
        cv::bilateralFilter(alpha, smooth, 5, 0.08, 2);
        smooth.copyTo(alpha, Mw);
    }
    cv::Mat G = m.gain(wr), O = m.offset(wr);
    double amax = 0;
    for (int y = 0; y < wr.height; ++y)
        for (int x = 0; x < wr.width; ++x) {
            if (!Mw.at<uchar>(y, x)) continue;
            auto a = alpha.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) {
                if (a[c] <= o.alphaMin) continue;
                G.at<cv::Vec3f>(y, x)[c] = 1 - a[c];
                O.at<cv::Vec3f>(y, x)[c] = float(a[c] * W[c]);
                amax = std::max(amax, (double)a[c]);
            }
        }
    fprintf(stderr, "  thin strokes: colour %.1f/%.1f/%.1f (BGR), max opacity %.3f\n", W[0], W[1], W[2], amax);
}

struct TonalFit {
    cv::Mat gain, offset, mark;
    std::vector<float> n, sp, si, spp, spi, sii;
    float ref;

    double misfit(int i, cv::Vec3f g, cv::Vec3f t) const {
        double e = 0;
        for (int j = 0; j < 3; ++j) {
            double c = t[j] - ref + g[j] * ref;
            int k = 3 * i + j;
            e += sii[k] - 2 * g[j] * spi[k] - 2 * c * si[k] + g[j] * g[j] * spp[k] + 2 * g[j] * c * sp[k] + n[i] * c * c;
        }
        return e;
    }
};

// Fits page = gain * paper + offset per pixel and channel, where paper(k) gives page k's paper level,
// over what page content leaves clear: marks darker than the paper beyond those every page shares are
// content. A pixel is marked where the fit departs from "no watermark" by more than its own noise.
static TonalFit fitTonal(const std::vector<std::string>& files, const std::vector<Xform>& xf, cv::Rect roi,
                         const std::function<cv::Vec3f(size_t, const cv::Mat&, const cv::Mat&)>& paper,
                         const cv::Mat& shared, float ref, const Options& o) {
    const int px = roi.area();
    std::vector<float> n(px, 0), sp(3 * px, 0), si(3 * px, 0), spp(3 * px, 0), spi(3 * px, 0), sii(3 * px, 0);
    for (size_t k = 0; k < files.size(); ++k) {
        if (!xf[k].present) continue;
        cv::Mat page = readImage(files[k], cv::IMREAD_COLOR), gray, c, v, cg, ink, inkV;
        cv::cvtColor(page, gray, cv::COLOR_BGR2GRAY);
        warpToCanon(page, xf[k], roi, c, v);
        warpToCanon(inkContrast(gray, o.inkSize), xf[k], roi, ink, inkV);
        cv::cvtColor(c, cg, cv::COLOR_BGR2GRAY);
        cv::Mat clear = v & ~((ink - shared > o.inkLevel) & (cg < paperLevel(gray) - o.inkLevel));
        cv::Vec3f pk = paper(k, c, clear);
        cv::parallel_for_(cv::Range(0, roi.height), [&](const cv::Range& r) {
            for (int y = r.start; y < r.end; ++y)
                for (int x = 0; x < roi.width; ++x) {
                    if (!clear.at<uchar>(y, x)) continue;
                    auto val = c.at<cv::Vec3b>(y, x);
                    int i = y * roi.width + x;
                    n[i] += 1;
                    for (int j = 0; j < 3; ++j) {
                        float u = pk[j] - ref, w = val[j] - ref;
                        sp[3 * i + j] += u; si[3 * i + j] += w; spp[3 * i + j] += u * u; spi[3 * i + j] += u * w; sii[3 * i + j] += w * w;
                    }
                }
        });
    }
    // A layer's gain is shared by its neighbouring pixels, its tone is not: the slope is pooled over a
    // window, which keeps the paper's own texture from being fitted away, the offset is per pixel.
    TonalFit f;
    f.gain = cv::Mat(roi.size(), CV_32FC3, cv::Scalar::all(1));
    f.offset = cv::Mat(roi.size(), CV_32FC3, cv::Scalar::all(0));
    f.mark = cv::Mat(roi.size(), CV_8U, cv::Scalar(0));
    auto solve = [&](const cv::Mat& within) {
        auto pool = [&](std::vector<float>& v, int ch) {
            cv::Mat in = cv::Mat(roi.size(), CV_32FC(ch), v.data()).clone(), out;
            if (!within.empty()) in.setTo(0, ~within);
            cv::boxFilter(in, out, -1, {7, 7}, {-1, -1}, false, cv::BORDER_REFLECT);
            return out;
        };
        cv::Mat N = pool(n, 1), SP = pool(sp, 3), SI = pool(si, 3), SPP = pool(spp, 3), SPI = pool(spi, 3), SII = pool(sii, 3);
        cv::parallel_for_(cv::Range(0, roi.height), [&](const cv::Range& r) {
            for (int y = r.start; y < r.end; ++y)
                for (int x = 0; x < roi.width; ++x) {
                    int i = y * roi.width + x;
                    double nn = n[i], np = N.at<float>(y, x);
                    if (nn < 8 || (!within.empty() && !within.at<uchar>(y, x))) continue;
                    double gm = 0, em = 0, seg = 0, see = 0;
                    cv::Vec3f gg, tt;
                    bool spread = true;
                    for (int j = 0; j < 3 && spread; ++j) {
                        double mu = SP.at<cv::Vec3f>(y, x)[j] / np, mi = SI.at<cv::Vec3f>(y, x)[j] / np;
                        double vu = SPP.at<cv::Vec3f>(y, x)[j] / np - mu * mu, cov = SPI.at<cv::Vec3f>(y, x)[j] / np - mu * mi;
                        double vi = SII.at<cv::Vec3f>(y, x)[j] / np - mi * mi;
                        if (vu < 100) { spread = false; break; }
                        double gj = cov / vu, rj = std::max(vi - gj * cov, 1.0);
                        double pu = sp[3 * i + j] / nn, pi = si[3 * i + j] / nn;
                        gg[j] = (float)gj;
                        tt[j] = float(pi + ref - gj * (pu + ref));
                        gm += gj / 3;
                        em += (pi - pu) / 3;
                        seg += std::sqrt(rj / (np * vu)) / 3;
                        see += std::sqrt(std::max(sii[3 * i + j] / nn - pi * pi, 1.0) / nn) / 3;
                    }
                    if (!spread) {
                        if (!within.empty()) f.mark.at<uchar>(y, x) = 0;
                        continue;
                    }
                    f.gain.at<cv::Vec3f>(y, x) = gg;
                    f.offset.at<cv::Vec3f>(y, x) = tt;
                    bool departs = (std::fabs(gm - 1) > 3 * seg && std::fabs(gm - 1) > 0.03) || (std::fabs(em) > 3 * see && std::fabs(em) > 3);
                    if (within.empty() && departs && gm > 0.3 && gm < 1.05) f.mark.at<uchar>(y, x) = 255;
                }
        });
    };
    solve(cv::Mat());
    cv::morphologyEx(f.mark, f.mark, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    // The window reaches past the watermark's edge; a pixel there is kept only if its own observations
    // follow the watermark's gain better than they follow the paper.
    for (int pass = 0; pass < 2; ++pass) {
        solve(f.mark.clone());
        for (int y = 0; y < roi.height; ++y)
            for (int x = 0; x < roi.width; ++x) {
                if (!f.mark.at<uchar>(y, x)) continue;
                int i = y * roi.width + x;
                double nn = n[i], in = 0, out = 0;
                for (int j = 0; j < 3; ++j) {
                    double mu = sp[3 * i + j] / nn, mi = si[3 * i + j] / nn;
                    double vu = spp[3 * i + j] / nn - mu * mu, cov = spi[3 * i + j] / nn - mu * mi;
                    double g = f.gain.at<cv::Vec3f>(y, x)[j];
                    in += g * g * vu - 2 * g * cov;
                    out += vu - 2 * cov;
                }
                if (in >= out) f.mark.at<uchar>(y, x) = 0;
            }
    }
    solve(f.mark.clone());
    f.gain.setTo(cv::Scalar::all(1), ~f.mark);
    f.offset.setTo(cv::Scalar::all(0), ~f.mark);
    f.n = std::move(n); f.sp = std::move(sp); f.si = std::move(si);
    f.spp = std::move(spp); f.spi = std::move(spi); f.sii = std::move(sii);
    f.ref = ref;
    return f;
}

// Where a watermark never moves and pages differ in paper tone (dark covers among white sheets, scans
// of differing brightness), a pixel's tone follows its page's paper through the watermark by exactly
// the watermark's gain, so gain and offset are measured directly. The watermark is what departs from
// "no watermark" and is connected to what all pages visibly agree on; each page's paper is then
// measured just around it.
static void estimateTonal(const std::vector<std::string>& files, const std::vector<cv::Mat>& grays,
                          const std::vector<Xform>& xf, const cv::Mat& agree, const cv::Mat& shared, Model& m,
                          const Options& o) {
    if (m.align) return;
    std::vector<double> P;
    std::vector<std::pair<double, size_t>> byTone;
    for (size_t k = 0; k < grays.size(); ++k)
        if (xf[k].present) {
            P.push_back(paperLevel(grays[k]));
            byTone.push_back({P.back(), k});
        }
    double mp = 0, vp = 0;
    for (double p : P) mp += p / P.size();
    for (double p : P) vp += (p - mp) * (p - mp) / P.size();
    if (P.size() < 8 || vp < 100) return;

    const cv::Rect all(0, 0, m.frame.width, m.frame.height);
    auto pageMedian = [&](size_t k, const cv::Mat&, const cv::Mat&) {
        cv::Mat page = readImage(files[k], cv::IMREAD_COLOR), ch[3];
        cv::split(page, ch);
        return cv::Vec3f(paperLevel(ch[0]), paperLevel(ch[1]), paperLevel(ch[2]));
    };
    TonalFit first = fitTonal(files, xf, all, pageMedian, shared, (float)mp, o);
    cv::Mat mark = componentsTouching(first.mark, agree);
    if (!cv::countNonZero(mark)) return;

    cv::Mat near, ring;
    cv::dilate(mark, near, ellipse(9));
    cv::dilate(mark, ring, ellipse(41));
    ring &= ~near;
    cv::Rect roi = maskBounds(ring);
    cv::Mat ringRoi = ring(roi);
    auto localPaper = [&](size_t, const cv::Mat& c, const cv::Mat& clear) {
        cv::Vec3f p;
        for (int j = 0; j < 3; ++j) {
            std::vector<float> v;
            for (int y = 0; y < c.rows; ++y)
                for (int x = 0; x < c.cols; ++x)
                    if (ringRoi.at<uchar>(y, x) && clear.at<uchar>(y, x)) v.push_back(c.at<cv::Vec3b>(y, x)[j]);
            p[j] = v.empty() ? 0.f : quantile(v, 0.5);
        }
        return p;
    };
    TonalFit fit = fitTonal(files, xf, roi, localPaper, shared(roi), (float)mp, o);

    // Two halves of the pages, each with its share of every paper tone, must each find it on their own:
    // what a single page brings (a sheet's edge, a cover's print) is not the watermark.
    std::sort(byTone.begin(), byTone.end());
    std::vector<Xform> half[2] = {xf, xf};
    for (size_t i = 0; i < byTone.size(); ++i) half[i % 2][byTone[i].second].present = false;
    TonalFit a = fitTonal(files, half[0], roi, localPaper, shared(roi), (float)mp, o),
             b = fitTonal(files, half[1], roi, localPaper, shared(roi), (float)mp, o);
    // A gain pooled over a window holds only for areas at least that wide: an area counts if both halves
    // find such a part of it. It replaces the other estimates, which measured the watermark through ink,
    // only if each half's fit predicts the other half's pages there far better than they do.
    cv::Mat seed, lab;
    cv::morphologyEx(fit.mark & a.mark & b.mark, seed, cv::MORPH_OPEN, cv::Mat::ones(7, 7, CV_8U));
    cv::Mat sel = componentsTouching(componentsTouching(fit.mark & near(roi), seed), agree(roi));
    int nl = cv::connectedComponents(sel, lab, 8, CV_32S);
    std::vector<double> held(nl, 0), prior(nl, 0);
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x) {
            int l = lab.at<int>(y, x), i = y * roi.width + x;
            if (!l) continue;
            cv::Vec3f g0 = m.gain(roi).at<cv::Vec3f>(y, x), t0 = m.offset(roi).at<cv::Vec3f>(y, x);
            held[l] += b.misfit(i, a.gain.at<cv::Vec3f>(y, x), a.offset.at<cv::Vec3f>(y, x)) +
                       a.misfit(i, b.gain.at<cv::Vec3f>(y, x), b.offset.at<cv::Vec3f>(y, x));
            prior[l] += a.misfit(i, g0, t0) + b.misfit(i, g0, t0);
        }
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x) {
            int l = lab.at<int>(y, x);
            if (l && held[l] >= 0.5 * prior[l]) sel.at<uchar>(y, x) = 0;
        }
    for (int y = 0; y < roi.height; ++y)
        for (int x = 0; x < roi.width; ++x) {
            if (!sel.at<uchar>(y, x)) continue;
            cv::Vec3f g = fit.gain.at<cv::Vec3f>(y, x), t = fit.offset.at<cv::Vec3f>(y, x);
            for (int j = 0; j < 3; ++j) {
                float gj = std::clamp(g[j], 0.05f, 1.f);
                t[j] += (g[j] - gj) * float(mp);
                g[j] = gj;
            }
            m.gain(roi).at<cv::Vec3f>(y, x) = g;
            m.offset(roi).at<cv::Vec3f>(y, x) = t;
        }
    fprintf(stderr, "  contrast fit over pages of differing paper tone: %.2f%% of frame\n",
            100.0 * cv::countNonZero(sel) / m.frame.area());
}

void estimateWatermark(const std::vector<std::string>& files, const std::vector<cv::Mat>& grays,
                       const std::vector<Xform>& xf, Model& m, const Options& o) {
    int npres = 0;
    for (auto& x : xf) npres += x.present;
    std::vector<cv::Mat> inkInv(grays.size());
    cv::parallel_for_(cv::Range(0, (int)grays.size()), [&](const cv::Range& r) {
        for (int k = r.start; k < r.end; ++k) inkInv[k] = 255 - inkContrast(grays[k], o.inkSize);
    });
    cv::Mat med, cnt, spread, inkMed, inkMin, icnt;
    stackMedian(grays, xf, 1, m.frame, med, cnt, nullptr, &spread);
    stackMedian(inkInv, xf, 1, m.frame, inkMed, icnt, &inkMin);
    inkInv.clear();
    cv::Mat ink = 255 - inkMin, valid = cnt >= std::max(2, npres / 2);

    float paper;
    cv::Mat agree = watermarkAgreement(med, spread, valid, o.strokeMax, paper), core, broadAgree = agree.clone();
    {
        // Where what lies under the watermark changes from page to page (a book's edges, the next page,
        // the dark around a sheet), pages agree only on how far it departs from their own paper, and
        // only pages that show paper there have a say.
        std::vector<cv::Mat> dep(grays.size());
        cv::parallel_for_(cv::Range(0, (int)grays.size()), [&](const cv::Range& r) {
            for (int k = r.start; k < r.end; ++k) {
                cv::Mat bg, d;
                cv::medianBlur(grays[k], bg, 41);
                cv::subtract(grays[k], bg, d, cv::noArray(), CV_16S);
                cv::Mat(d + 128).convertTo(dep[k], CV_8U);
                cv::max(dep[k], 1, dep[k]);
                dep[k].setTo(0, bg < paperLevel(grays[k]) - 40);
            }
        });
        cv::Mat dmed, dcnt, dspread;
        float flat;
        stackMedian(dep, xf, 1, m.frame, dmed, dcnt, nullptr, &dspread, true);
        // It only carries on a watermark already seen, mark by mark, not content that is merely
        // laid out alike on every page.
        cv::Mat more = watermarkAgreement(dmed, dspread, dcnt >= std::max(2, npres / 4), o.strokeMax, flat, 0.4), near;
        cv::dilate(agree | more, near, ellipse(81));
        agree |= more & componentsTouching(near, agree);
    }
    cv::erode(agree, core, ellipse(o.strokeMax));
    cv::Mat seed = componentsTouching(agree, core), thin = agree & ~seed;
    bool bright = cv::countNonZero(thin) && cv::mean(med, thin)[0] > paper;
    cv::Mat smooth, nearThin;
    cv::morphologyEx(med, smooth, bright ? cv::MORPH_OPEN : cv::MORPH_CLOSE, ellipse(o.strokeMax));
    cv::Mat lift = bright ? med - smooth : smooth - med;
    cv::dilate(thin, nearThin, ellipse(15));
    thin = componentsTouching(((lift > o.detectThresh) & valid & nearThin) | thin, thin);
    double contrast = 0;
    cv::Mat R = cv::countNonZero(seed) ? flatRegion(seed, ink, valid, thin, o, contrast) : cv::Mat::zeros(m.frame, CV_8U);
    if (cv::countNonZero(R)) {
        cv::Mat nearR;
        cv::dilate(R, nearR, ellipse(4 * o.block + 1));
        cv::Mat attached = componentsTouching(thin, nearR);
        R |= attached;
        thin &= ~attached;
        cv::dilate(R, R, ellipse(7));
        R &= valid;
    }
    cv::Mat M, excl;
    cv::dilate(thin, M, ellipse(5));
    M &= ~R;
    cv::dilate(thin, excl, ellipse(9));
    excl |= R;
    fprintf(stderr, "estimating: paper %.0f, ink contrast %.0f; thin strokes %.2f%%, flat layers %.2f%% of frame\n",
            paper, contrast, 100.0 * cv::countNonZero(M) / m.frame.area(), 100.0 * cv::countNonZero(R) / m.frame.area());

    m.gain = cv::Mat(m.frame, CV_32FC3, cv::Scalar::all(1));
    m.offset = cv::Mat(m.frame, CV_32FC3, cv::Scalar::all(0));
    if (cv::countNonZero(R)) estimateFlat(files, xf, R, contrast, o.inkSize, m);
    cv::Mat shared;
    cv::Mat(255 - inkMed).convertTo(shared, CV_8U);
    if (cv::countNonZero(M)) estimateThin(files, xf, M, excl, shared, bright ? 255 : 0, m, o);
    estimateTonal(files, grays, xf, broadAgree, shared, m, o);
}
