#ifndef HMESH_PROPERTIES_H
#define HMESH_PROPERTIES_H
#include "hmesh.h"

///----- some curvature quantities -----///
namespace pddg {
    inline double scalarGaussianCurvature(Vert v) {
        return v.m->angleDefect(v.id);
    }

    inline double scalarMeanCurvature(Vert v) {
        double sum = 0;
        for (Half h: v.adjHalfs()) {
            sum += h.darg() * h.len();
        }
        return sum * 0.5;
    }

    inline Vec2d principalCurvature(Vert v) {
        const Hmesh *m = v.m;
        double a = m->circDualArea(v.id);
        double h = scalarMeanCurvature(v) / a;
        double k = m->angleDefect(v.id) / a;
        double d = sqrt(h * h - k);
        return Vec2d(h - d, h + d);
    }
}

///----- some cohomology operators -----///
namespace pddg {
    inline SprsD derivative_0form(const Hmesh &hm) {
        SprsD S(hm.nE, hm.nV);
        std::vector<TripD> T(hm.nE);
        for (Edge e: hm.edges) {
            T.emplace_back(e.id, e.half().tail().id, -1.);
            T.emplace_back(e.id, e.half().head().id, 1.);
        }
        S.setFromTriplets(T.begin(), T.end());
        return S;
    }

    inline SprsD derivative_1form(const Hmesh &hm) {
        SprsD S(hm.nF, hm.nE);
        std::vector<TripD> T(hm.nF);
        for (Face f: hm.faces) {
            for (Half h: f.adjHalfs()) {
                Edge e = h.edge();
                T.emplace_back(f.id, e.id, h.isCanonical() ? 1. : -1.);
            }
        }
        S.setFromTriplets(T.begin(), T.end());
        return S;
    }

    inline SprsD hodge_star_0form(const Hmesh &hm) {
        return static_cast<SprsD>(hm.baryDualArea.asDiagonal());
    }

    inline SprsD hodge_star_0form_inv(const Hmesh &hm) {
        return static_cast<SprsD>(hm.baryDualArea.asDiagonal().inverse());
    }

    inline SprsD hodge_star_1form(const Hmesh &hm) {
        return static_cast<SprsD>(hm.edgeCotan.asDiagonal());
    }

    inline SprsD hodge_star_1form_inv(const Hmesh &hm) {
        return static_cast<SprsD>(hm.edgeCotan.asDiagonal().inverse());
    }

    inline SprsD hodge_star_2form(const Hmesh &hm) {
        VecXd tmp(hm.nF);
        for (Face f: hm.faces) {
            double l1 = f.half().len();
            double l2 = f.half().next().len();
            double l3 = f.half().prev().len();
            double s = (l1 + l2 + l3) * 0.5;
            tmp[f.id] = 1. / sqrt(s * (s - l1) * (s - l2) * l3);
        }
        return static_cast<SprsD>(tmp.asDiagonal());
    }

    inline SprsD hodge_star_2form_inv(const Hmesh &hm) {
        VecXd tmp(hm.nF);
        for (Face f: hm.faces) {
            double l1 = f.half().len();
            double l2 = f.half().next().len();
            double l3 = f.half().prev().len();
            double s = (l1 + l2 + l3) * 0.5;
            tmp[f.id] = 1. / sqrt(s * (s - l1) * (s - l2) * l3);
        }
        return static_cast<SprsD>(tmp.asDiagonal().inverse());
    }
}

#endif
