#include "dewatermark.hpp"

#include <algorithm>
#include <cstdio>

const char* anchorName(int a) {
    static const char* n[] = {"top-left", "top-right", "bottom-left", "bottom-right", "center"};
    return n[a];
}

cv::Mat highpass(const cv::Mat& src, double sigma) {
    cv::Mat f, b;
    src.convertTo(f, CV_32F);
    cv::GaussianBlur(f, b, {0, 0}, sigma);
    return f - b;
}

static cv::Point2d anchorPoint(int anchor, cv::Size s) {
    switch (anchor) {
        case TL: return {0, 0};
        case TR: return {double(s.width), 0};
        case BL: return {0, double(s.height)};
        case BR: return {double(s.width), double(s.height)};
        default: return {s.width / 2.0, s.height / 2.0};
    }
}

Xform initialXform(const Model& m, cv::Size page) {
    Xform x;
    x.a = m.propScale ? double(page.width) / m.refSize.width : 1.0;
    cv::Point2d ap = anchorPoint(m.anchor, page), ar = anchorPoint(m.anchor, m.refSize);
    x.bx = ap.x - x.a * ar.x + x.a * m.origin.x;
    x.by = ap.y - x.a * ar.y + x.a * m.origin.y;
    return x;
}

double paperLevel(const cv::Mat& ch) {
    int hist[256] = {0};
    for (int y = 0; y < ch.rows; ++y) {
        const uchar* p = ch.ptr<uchar>(y);
        for (int x = 0; x < ch.cols; ++x) hist[p[x]]++;
    }
    double half = ch.total() / 2.0, acc = 0;
    for (int v = 0; v < 256; ++v) if ((acc += hist[v]) >= half) return v;
    return 255;
}

void stackMedian(const std::vector<cv::Mat>& g, const std::vector<Xform>& xf, double div, cv::Size frame,
                 cv::Mat& med, cv::Mat& count, cv::Mat* darkest, cv::Mat* spread, bool skipZero) {
    std::vector<int> use;
    for (size_t k = 0; k < g.size(); ++k) if (xf[k].present) use.push_back((int)k);
    med.create(frame, CV_32F);
    count.create(frame, CV_32S);
    if (darkest) darkest->create(frame, CV_32F);
    if (spread) spread->create(frame, CV_32F);
    cv::parallel_for_(cv::Range(0, frame.height), [&](const cv::Range& r) {
        std::vector<float> v;
        v.reserve(use.size());
        for (int y = r.start; y < r.end; ++y) {
            float* dr = darkest ? darkest->ptr<float>(y) : nullptr;
            float* mr = med.ptr<float>(y);
            int* cr = count.ptr<int>(y);
            for (int x = 0; x < frame.width; ++x) {
                v.clear();
                float d = 255;
                for (int k : use) {
                    const Xform& t = xf[k];
                    float s;
                    if (!sampleAt(g[k], t.a * x + t.bx / div, t.a * y + t.by / div, s) || (skipZero && s < 0.5f)) continue;
                    v.push_back(s);
                    d = std::min(d, s);
                }
                cr[x] = (int)v.size();
                if (dr) dr[x] = d;
                if (v.empty()) { mr[x] = 0; if (spread) spread->at<float>(y, x) = 255; continue; }
                size_t h = v.size() / 2;
                std::nth_element(v.begin(), v.begin() + h, v.end());
                mr[x] = v[h];
                if (spread) {
                    for (float& e : v) e = std::fabs(e - mr[x]);
                    size_t q = v.size() / 2;
                    std::nth_element(v.begin(), v.begin() + q, v.end());
                    spread->at<float>(y, x) = v[q];
                }
            }
        }
    });
}

cv::Mat watermarkAgreement(const cv::Mat& med, const cv::Mat& spread, const cv::Mat& valid, int wide, float& paper,
                           double tolerance) {
    // Paper is the most common tone: content that every page has in much the same place (ruled staves)
    // would pull a median off it.
    std::vector<double> hist(256, 0);
    for (int y = 0; y < med.rows; ++y)
        for (int x = 0; x < med.cols; ++x)
            if (valid.at<uchar>(y, x)) hist[std::clamp((int)std::lround(med.at<float>(y, x)), 0, 255)]++;
    int best = -1;
    double top = 0;
    for (int t = 0; t < 256; ++t) {
        double h = 0;
        for (int u = std::max(0, t - 2); u <= std::min(255, t + 2); ++u) h += hist[u];
        if (h > top) { top = h; best = t; }
    }
    if (best < 0) { paper = 255; return cv::Mat::zeros(med.size(), CV_8U); }
    paper = (float)best;
    cv::Mat d = cv::abs(med - paper), agree = valid & (d > 4) & (spread < tolerance * d);
    cv::morphologyEx(agree, agree, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    // The scan's background around the sheet is the same on every page too, but it lies at the
    // image border: wide there, or confined to a band along it. An overlay sits on the page.
    cv::Mat core, lab;
    cv::erode(agree, core, cv::getStructuringElement(cv::MORPH_ELLIPSE, {wide, wide}));
    int n = cv::connectedComponents(agree, lab, 8);
    std::vector<char> isWide(n, 0), nearBorder(n, 0), atBorder(n, 0), inland(n, 0);
    for (int y = 0; y < agree.rows; ++y)
        for (int x = 0; x < agree.cols; ++x)
            if (int l = lab.at<int>(y, x)) {
                int edge = std::min({x, y, agree.cols - 1 - x, agree.rows - 1 - y});
                isWide[l] |= core.at<uchar>(y, x) != 0;
                nearBorder[l] |= edge < wide;
                atBorder[l] |= edge == 0;
                inland[l] |= edge >= wide;
            }
    for (int y = 0; y < agree.rows; ++y)
        for (int x = 0; x < agree.cols; ++x)
            if (int l = lab.at<int>(y, x); l && ((isWide[l] && nearBorder[l]) || (atBorder[l] && !inland[l])))
                agree.at<uchar>(y, x) = 0;
    return agree;
}

static cv::Mat makeTemplate(const cv::Mat& med, const cv::Mat& agree, float paper, double sigma) {
    cv::Mat wm(med.size(), CV_32F, cv::Scalar(paper));
    med.copyTo(wm, agree);
    cv::Mat t = highpass(wm, sigma), near;
    int k = 2 * (int)std::ceil(2 * sigma) + 1;
    cv::dilate(agree, near, cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k}));
    t.setTo(0, ~near);
    return t;
}

struct Shift { double dx, dy, ncc; };

static double parabola(double l, double c, double r) {
    double d = l - 2 * c + r;
    return d < 0 ? 0.5 * (l - r) / d : 0;
}

static Shift matchAt(const cv::Mat& H, const cv::Mat& T, cv::Rect roi, double a, double bx, double by,
                     int r, bool subpix) {
    cv::Mat M(cv::Matx23d(a, 0, bx + a * (roi.x - r), 0, a, by + a * (roi.y - r)));
    cv::Mat dst, res, mask;
    cv::warpAffine(H, dst, M, cv::Size(roi.width + 2 * r, roi.height + 2 * r),
                   cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);
    cv::Mat(cv::abs(T(roi)) > 1).convertTo(mask, CV_32F, 1 / 255.0);
    cv::matchTemplate(dst, T(roi), res, cv::TM_CCORR_NORMED, mask);
    cv::patchNaNs(res, -1);
    cv::Point p;
    double mx;
    cv::minMaxLoc(res, nullptr, &mx, nullptr, &p);
    double dx = p.x, dy = p.y;
    if (subpix) {
        if (p.x > 0 && p.x < res.cols - 1)
            dx += parabola(res.at<float>(p.y, p.x - 1), mx, res.at<float>(p.y, p.x + 1));
        if (p.y > 0 && p.y < res.rows - 1)
            dy += parabola(res.at<float>(p.y - 1, p.x), mx, res.at<float>(p.y + 1, p.x));
    }
    return {a * (dx - r), a * (dy - r), mx};
}

static void rescaleAbout(Xform& x, double a, cv::Point2d centre) {
    double ix = x.a * centre.x + x.bx, iy = x.a * centre.y + x.by;
    x.a = a;
    x.bx = ix - a * centre.x;
    x.by = iy - a * centre.y;
}

static Xform searchCoarse(const cv::Mat& hpSmall, const cv::Mat& tmpl, Xform x, bool scaleVaries,
                          const Options& o) {
    const double c = o.coarse;
    int r = std::max(2, (int)std::ceil(o.search / c));
    cv::Rect roi = maskBounds(tmpl != 0);
    if (roi.empty()) roi = cv::Rect(0, 0, tmpl.cols, tmpl.rows);
    cv::Point2d centre(c * (roi.x + roi.width / 2.0), c * (roi.y + roi.height / 2.0));
    std::vector<double> scales{x.a};
    if (scaleVaries) {
        scales.clear();
        for (int i = -6; i <= 6; ++i) scales.push_back(x.a * (1 + i * o.scaleRange / 6));
    }
    Xform best = x;
    best.ncc = -2;
    for (double a : scales) {
        Xform t = x;
        rescaleAbout(t, a, centre);
        Shift s = matchAt(hpSmall, tmpl, roi, t.a, t.bx / c, t.by / c, r, false);
        if (s.ncc > best.ncc) { best = t; best.bx += c * s.dx; best.by += c * s.dy; best.ncc = s.ncc; }
    }
    return best;
}

static Xform searchFine(const cv::Mat& hpFull, const Model& m, Xform x, const Options& o) {
    int r = o.coarse + 2;
    cv::Rect roi = m.fineRoi;
    cv::Point2d centre(roi.x + roi.width / 2.0, roi.y + roi.height / 2.0);
    auto at = [&](double a) {
        Xform t = x;
        rescaleAbout(t, a, centre);
        Shift s = matchAt(hpFull, m.tmplFull, roi, t.a, t.bx, t.by, r, true);
        t.bx += s.dx; t.by += s.dy; t.ncc = s.ncc;
        return t;
    };
    if (!m.propScale) return at(1.0);
    const double step = o.scaleRange / 24;
    std::vector<Xform> c;
    for (int i = -4; i <= 4; ++i) c.push_back(at(x.a * (1 + i * step)));
    int b = 0;
    for (int i = 1; i < (int)c.size(); ++i) if (c[i].ncc > c[b].ncc) b = i;
    double off = (b > 0 && b < (int)c.size() - 1) ? parabola(c[b - 1].ncc, c[b].ncc, c[b + 1].ncc) : 0;
    Xform t = at(x.a * (1 + (b - 4 + off) * step));
    t = at(t.a);
    if (std::fabs(t.a - 1) < 1e-4) t.a = 1;
    return t;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

static double updatePresence(std::vector<Xform>& xf, const Options& o) {
    std::vector<double> n;
    for (auto& x : xf) n.push_back(x.ncc);
    double med = median(n);
    for (auto& x : xf) x.present = x.ncc >= o.presentRatio * med;
    return med;
}

static void roundPlacement(Xform& x) {
    x.bx = std::round(x.bx);
    x.by = std::round(x.by);
}

static cv::Rect pickRoi(const cv::Mat& tmpl, int side) {
    cv::Rect sup = maskBounds(tmpl != 0);
    if (sup.empty()) sup = cv::Rect(0, 0, tmpl.cols, tmpl.rows);
    int w = std::min(side, sup.width), h = std::min(side, sup.height);
    cv::Mat e = cv::abs(tmpl), s;
    cv::integral(e, s, CV_64F);
    double best = -1;
    cv::Rect r(sup.x, sup.y, w, h);
    int step = std::max(8, side / 16);
    for (int y = sup.y; y + h <= sup.y + sup.height; y += step)
        for (int x = sup.x; x + w <= sup.x + sup.width; x += step) {
            double v = s.at<double>(y + h, x + w) - s.at<double>(y, x + w) - s.at<double>(y + h, x) + s.at<double>(y, x);
            if (v > best) { best = v; r = {x, y, w, h}; }
        }
    return r;
}

static cv::Mat matchImage(const cv::Mat& gray, double inkClamp) {
    cv::Mat out;
    cv::max(gray, paperLevel(gray) - inkClamp, out);
    return out;
}

static cv::Mat downsample(const cv::Mat& g, int c) {
    cv::Mat s;
    cv::resize(g, s, cv::Size(g.cols / c, g.rows / c), 0, 0, cv::INTER_AREA);
    return s;
}

void registerSet(const std::vector<cv::Mat>& grays, Model& m, std::vector<Xform>& xf, const Options& o) {
    const int n = (int)grays.size(), c = o.coarse;
    std::vector<cv::Mat> hpS(n);
    auto prepare = [&] {
        cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& r) {
            for (int k = r.start; k < r.end; ++k)
                hpS[k] = highpass(downsample(matchImage(grays[k], m.inkClamp), c), o.hpSigma / c);
        });
    };
    m.refSize = grays[0].size();
    m.origin = {0, 0};
    m.frame = m.refSize;
    cv::Size frameS(m.frame.width / c, m.frame.height / c);
    bool sizesVary = false;
    for (auto& g : grays) sizesVary |= std::fabs(double(g.cols) / m.refSize.width - 1) > 0.005;

    auto fullAgreement = [&](const std::vector<Xform>& x, cv::Mat& med, cv::Mat& valid, float& paper) {
        cv::Mat cnt, spread;
        stackMedian(grays, x, 1, m.frame, med, cnt, nullptr, &spread);
        valid = cnt >= std::max(2, n / 2);
        return watermarkAgreement(med, spread, valid, o.strokeMax, paper);
    };
    auto coarseTemplate = [&](const std::vector<Xform>& x) {
        cv::Mat med, valid, wm(m.frame, CV_32F), wmSmall, agreeSmall;
        float paper;
        cv::Mat agree = fullAgreement(x, med, valid, paper);
        wm.setTo(paper);
        med.copyTo(wm, agree);
        cv::resize(wm, wmSmall, frameS, 0, 0, cv::INTER_AREA);
        cv::resize(agree, agreeSmall, frameS, 0, 0, cv::INTER_AREA);
        return makeTemplate(wmSmall, agreeSmall > 0, paper, o.hpSigma / c);
    };
    auto refineAll = [&](std::vector<Xform>& x, const cv::Mat& t, bool prop) {
        cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& r) {
            for (int k = r.start; k < r.end; ++k) {
                bool pres = x[k].present;
                x[k] = searchCoarse(hpS[k], t, x[k], prop, o);
                x[k].present = pres;
            }
        });
        updatePresence(x, o);
    };
    struct Fit { std::vector<Xform> x; double spread = 1e18, shift = 1e18; };
    auto residuals = [&](const std::vector<Xform>& init, std::vector<Xform> x, const cv::Mat& t) {
        cv::Rect sup = maskBounds(t != 0);
        cv::Point2d at(c * (sup.x + sup.width / 2.0), c * (sup.y + sup.height / 2.0));
        const double limit = c * (std::ceil(o.search / c) - 1);
        std::vector<double> rx, ry, dev;
        for (int k = 0; k < n; ++k) {
            double dx = (x[k].a - init[k].a) * at.x + x[k].bx - init[k].bx;
            double dy = (x[k].a - init[k].a) * at.y + x[k].by - init[k].by;
            if (std::fabs(dx) < limit && std::fabs(dy) < limit) { rx.push_back(dx); ry.push_back(dy); }
        }
        Fit f;
        f.x = std::move(x);
        if ((int)rx.size() > n / 2) {
            double mx = median(rx), my = median(ry);
            for (size_t k = 0; k < rx.size(); ++k) dev.push_back(std::hypot(rx[k] - mx, ry[k] - my));
            f.spread = median(dev);
            f.shift = std::hypot(mx, my);
        }
        return f;
    };
    // Ink darker than anything the pages agree on as they are is clamped, so content weighs no more
    // in matching than the watermark does.
    auto clampInk = [&](const std::vector<Xform>& x) {
        cv::Mat med, valid;
        float paper;
        cv::Mat agree = fullAgreement(x, med, valid, paper);
        double lo = paper;
        if (cv::countNonZero(agree)) cv::minMaxLoc(med, &lo, nullptr, nullptr, nullptr, agree);
        m.inkClamp = std::max(8.0, paper - std::min<double>(lo, paper) + 8);
        prepare();
    };
    {
        std::vector<Xform> x(n);
        for (int k = 0; k < n; ++k) x[k] = initialXform(m, grays[k].size());
        clampInk(x);
    }
    const cv::Mat refTemplate = hpS[0](cv::Rect(0, 0, std::min(frameS.width, hpS[0].cols), std::min(frameS.height, hpS[0].rows)));
    auto evaluate = [&](int anchor, bool prop) {
        Model h = m;
        h.anchor = anchor;
        h.propScale = prop;
        std::vector<Xform> init(n);
        for (int k = 0; k < n; ++k) init[k] = initialXform(h, grays[k].size());
        std::vector<Xform> x = init;
        cv::Mat t = coarseTemplate(x);
        refineAll(x, t, prop);
        Fit best = residuals(init, x, t);
        x = init;
        refineAll(x, refTemplate, prop);
        t = coarseTemplate(x);
        refineAll(x, t, prop);
        Fit boot = residuals(init, x, t);
        if (boot.spread < best.spread) best = boot;
        fprintf(stderr, "  %-12s %-6s residual spread %5.1f px, common shift %5.1f px\n", anchorName(anchor),
                prop ? "scaled" : "fixed", std::min(best.spread, 1e4), std::min(best.shift, 1e4));
        return best;
    };
    auto choose = [&](bool prop, Fit best) {
        m.anchor = TL;
        for (int an = best.x.empty() ? 0 : 1; an < NUM_ANCHORS; ++an) {
            Fit f = evaluate(an, prop);
            if (f.spread < best.spread - 0.5) { best = f; m.anchor = an; }
        }
        m.propScale = prop;
        return best;
    };

    m.anchor = TL;
    m.propScale = false;
    Fit fit = evaluate(TL, false);
    bool asIs = fit.spread <= 0.5;
    if (asIs) {
        // A watermark scaled about its own centre stays put there; scaling shows only in how much better
        // the pages then match it.
        Fit scaled = evaluate(TL, true);
        std::vector<double> n0, n1;
        for (int k = 0; k < n; ++k) { n0.push_back(fit.x[k].ncc); n1.push_back(scaled.x[k].ncc); }
        if (median(n1) > 1.2 * median(n0)) {
            asIs = false;
            fit = scaled;
            m.propScale = true;
        }
    } else {
        fit = choose(false, fit);
        if (sizesVary) {
            int fixedAnchor = m.anchor;
            Fit scaled = choose(true, Fit());
            if (scaled.spread < 0.5 * fit.spread - 0.5) fit = scaled;
            else { m.anchor = fixedAnchor; m.propScale = false; }
        }
    }
    if (asIs) {
        m.anchor = TL;
        m.propScale = false;
        for (int k = 0; k < n; ++k) fit.x[k] = initialXform(m, grays[k].size());
    }
    xf = fit.x;

    clampInk(xf);
    if (!asIs) refineAll(xf, coarseTemplate(xf), m.propScale);

    double minx = 0, miny = 0, maxx = m.refSize.width, maxy = m.refSize.height;
    for (int k = 0; k < n; ++k) {
        const Xform& x = xf[k];
        minx = std::min(minx, -x.bx / x.a);
        miny = std::min(miny, -x.by / x.a);
        maxx = std::max(maxx, (grays[k].cols - x.bx) / x.a);
        maxy = std::max(maxy, (grays[k].rows - x.by) / x.a);
    }
    m.origin = {(int)std::floor(minx / c) * c, (int)std::floor(miny / c) * c};
    m.frame = {(int)std::ceil((maxx - m.origin.x) / c) * c, (int)std::ceil((maxy - m.origin.y) / c) * c};
    for (auto& x : xf) { x.bx += x.a * m.origin.x; x.by += x.a * m.origin.y; }

    auto fullTemplate = [&] {
        cv::Mat med, valid;
        float paper;
        cv::Mat agree = fullAgreement(xf, med, valid, paper);
        return makeTemplate(med, agree, paper, o.fineSigma);
    };
    m.tmplFull = fullTemplate();
    m.fineRoi = pickRoi(m.tmplFull, o.fineRoi);
    if (asIs) {
        // Pages that agree where they are stay there: per-page matching of a faint mark would
        // only pick up noise from the content.
        cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& r) {
            for (int k = r.start; k < r.end; ++k)
                xf[k].ncc = matchAt(highpass(matchImage(grays[k], m.inkClamp), o.fineSigma), m.tmplFull, m.fineRoi,
                                    xf[k].a, xf[k].bx, xf[k].by, 0, false).ncc;
        });
    } else {
        for (int pass = 0; pass < 2; ++pass) {
            if (pass) {
                m.tmplFull = fullTemplate();
                m.fineRoi = pickRoi(m.tmplFull, o.fineRoi);
            }
            cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& r) {
                for (int k = r.start; k < r.end; ++k) {
                    bool pres = xf[k].present;
                    xf[k] = searchFine(highpass(matchImage(grays[k], m.inkClamp), o.fineSigma), m, xf[k], o);
                    if (!m.propScale) roundPlacement(xf[k]);
                    xf[k].present = pres;
                }
            });
        }
    }
    m.align = !asIs;
    m.medNcc = updatePresence(xf, o);
    if (!m.align) fprintf(stderr, "  -> pages used as they are (no alignment needed)\n");
    else fprintf(stderr, "  -> aligned%s, anchored %s\n", m.propScale ? " and scaled" : "", anchorName(m.anchor));
    m.tmplFull = fullTemplate();
    frameS = cv::Size(m.frame.width / c, m.frame.height / c);
    m.tmplSmall = coarseTemplate(xf);
}

// Scaled pages are first matched to what all pages agree on, which may include content they share in
// layout (ruled staves); once the watermark itself is known, each page is matched to its text line,
// at both ends to fix the scale as well. A page keeps the new placement only where it matches better.
WatermarkLine watermarkLine(const Model& m) {
    WatermarkLine L;
    std::vector<cv::Mat> g, t;
    cv::split(m.gain, g);
    cv::split(m.offset, t);
    cv::Mat effect = cv::Mat::zeros(m.frame, CV_32F);
    for (int c = 0; c < 3; ++c) effect += ((g[c] - 1) * 200 + t[c]) / 3;
    cv::Mat mark = cv::abs(effect) > 4, lab, stats, cent;
    cv::morphologyEx(mark, mark, cv::MORPH_CLOSE, cv::Mat::ones(9, 9, CV_8U));
    int n = cv::connectedComponentsWithStats(mark, lab, stats, cent, 8, CV_32S);
    int big = 0;
    for (int i = 1; i < n; ++i) if (!big || stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(big, cv::CC_STAT_AREA)) big = i;
    if (!big) return L;
    int top = stats.at<int>(big, cv::CC_STAT_TOP), bottom = top + stats.at<int>(big, cv::CC_STAT_HEIGHT);
    cv::Mat line = cv::Mat::zeros(m.frame, CV_8U);
    for (int i = 1; i < n; ++i) {
        double cy = cent.at<double>(i, 1);
        if (cy >= top && cy <= bottom && stats.at<int>(i, cv::CC_STAT_AREA) > 30) line |= lab == i;
    }
    cv::Rect box = maskBounds(line);
    box = cv::Rect(box.x - 6, box.y - 6, box.width + 12, box.height + 12) & cv::Rect(0, 0, m.frame.width, m.frame.height);
    if (box.width < 80) return L;
    cv::dilate(line, line, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat masked = cv::Mat::zeros(m.frame, CV_32F);
    effect.copyTo(masked, line);
    L.tmpl = highpass(masked(box), 3);
    L.box = box;
    return L;
}

static double lineFit(const cv::Mat& hp, const WatermarkLine& L, const Xform& x, cv::Point2d* ends) {
    const int R = 6, half = L.box.width / 2;
    double score = 0;
    for (int k = 0; k < 2; ++k) {
        cv::Rect part(k * half, 0, k ? L.box.width - half : half, L.box.height);
        cv::Mat M(cv::Matx23d(x.a, 0, x.a * (L.box.x + part.x - R) + x.bx, 0, x.a, x.a * (L.box.y - R) + x.by)), win, r;
        cv::warpAffine(hp, win, M, cv::Size(part.width + 2 * R, part.height + 2 * R), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
        cv::matchTemplate(win, L.tmpl(part), r, cv::TM_CCORR_NORMED);
        cv::patchNaNs(r, -1);
        score += r.at<float>(R, R);
        if (!ends) continue;
        cv::Point p;
        double mx;
        cv::minMaxLoc(r, nullptr, &mx, nullptr, &p);
        double dx = p.x, dy = p.y;
        if (p.x > 0 && p.x < r.cols - 1) dx += parabola(r.at<float>(p.y, p.x - 1), mx, r.at<float>(p.y, p.x + 1));
        if (p.y > 0 && p.y < r.rows - 1) dy += parabola(r.at<float>(p.y - 1, p.x), mx, r.at<float>(p.y + 1, p.x));
        ends[k] = {L.box.x + part.x + part.width / 2.0 + dx - R, L.box.y + L.box.height / 2.0 + dy - R};
    }
    return score;
}

Xform refineToLine(const cv::Mat& gray, const Model& m, const WatermarkLine& L, Xform x) {
    if (L.tmpl.empty() || !x.present) return x;
    cv::Mat hp = highpass(matchImage(gray, m.inkClamp), 3);
    Xform best = x;
    double before = lineFit(hp, L, x, nullptr);
    const int half = L.box.width / 2;
    const double U1 = L.box.x + half / 2.0, U2 = L.box.x + half + (L.box.width - half) / 2.0, V = L.box.y + L.box.height / 2.0;
    for (int it = 0; it < 3; ++it) {
        cv::Point2d e[2];
        lineFit(hp, L, x, e);
        Xform n = x;
        n.a = (x.a * e[1].x - x.a * e[0].x) / (U2 - U1);
        n.bx = x.a * e[0].x + x.bx - n.a * U1;
        n.by = x.a * (e[0].y + e[1].y) / 2 + x.by - n.a * V;
        if (std::fabs(n.a / x.a - 1) > 0.02) break;
        x = n;
    }
    if (lineFit(hp, L, x, nullptr) > before + 0.01) { best.a = x.a; best.bx = x.bx; best.by = x.by; }
    return best;
}

Xform registerPage(const cv::Mat& gray, const Model& m, const Options& o) {
    Xform x = initialXform(m, gray.size());
    cv::Mat g = matchImage(gray, m.inkClamp);
    if (!m.align) {
        x.ncc = matchAt(highpass(g, o.fineSigma), m.tmplFull, m.fineRoi, x.a, x.bx, x.by, 0, false).ncc;
        x.present = x.ncc >= o.presentRatio * m.medNcc;
        return x;
    }
    x = searchCoarse(highpass(downsample(g, o.coarse), o.hpSigma / o.coarse), m.tmplSmall, x, m.propScale, o);
    x = searchFine(highpass(g, o.fineSigma), m, x, o);
    if (!m.propScale) roundPlacement(x);
    x.present = x.ncc >= o.presentRatio * m.medNcc;
    if (m.propScale) x = refineToLine(gray, m, watermarkLine(m), x);
    return x;
}
