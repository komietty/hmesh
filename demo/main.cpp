#include <memory>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/principal_curvature.h>
#include <polyscope/polyscope.h>
#include <polyscope/curve_network.h>
#include <polyscope/surface_mesh.h>
#include "hmesh/hmesh.h"
#include "hmesh/properties.h"
#include "hmesh/tree_cotree.h"


using namespace pddg;

int main(int argc, char *argv[]) {
    polyscope::options::autocenterStructures = true;

    MatXd V;
    MatXi F;
    igl::readOBJ(argv[1], V, F);
    auto hmesh = std::make_unique<Hmesh>(V, F);
    auto hgen = std::make_unique<Hgen>(*hmesh);
    hgen->calcHomologyGens(false);

    polyscope::init();
    polyscope::view::bgColor = std::array<float, 4>{0.02, 0.02, 0.02, 1};
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
    auto ps = polyscope::registerSurfaceMesh("mesh", hmesh->pos, hmesh->idx);

    /*--- visuailize curvature ---*/


    ps->setSurfaceColor({0, 10./ 255., 27./ 255.});
    VecXd GC(hmesh->nV);
    VecXd MC(hmesh->nV);
    MatXd PC(hmesh->nV, 2);
    for (Vert v: hmesh->verts) {
        GC[v.id] = scalarGaussianCurvature(v);
        MC[v.id] = scalarMeanCurvature(v);
        PC.row(v.id) = principalCurvature(v).transpose();
    }
    ps->addVertexScalarQuantity("Gaussian Curvature", GC);
    ps->addVertexScalarQuantity("Mean Curvature", MC);
    ps->addVertexScalarQuantity("Principal Curvature 1", PC.col(0));
    ps->addVertexScalarQuantity("Principal Curvature 2", PC.col(1));

    { //--- another curvature definition by panozzo 2010 ---//
        MatXd HN;
        SprsD L, M, Minv;
        igl::cotmatrix(V, F, L);
        igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
        igl::invert_diag(M, Minv);
        HN = -Minv * (L * V);
        MatXd PD1, PD2;
        VecXd PV1, PV2;
        igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);
        VecXd H1 = HN.rowwise().norm();
        VecXd H2 = 0.5 * (PV1 + PV2);
        VecXd K = PV1.array() * PV2.array();
        ps->addVertexScalarQuantity("Gaussian Curvature panozzo", K);
    ps->addVertexScalarQuantity("Principal Curvature 1 panozzo", PV1);
    ps->addVertexScalarQuantity("Principal Curvature 2 panozzo", PV2);
    }

    ps->resetTransform();
    ps->setSmoothShade(true);

    /*--- visuailize face basis ---*/

    /*--- visuailize vert basis ---*/

    /*--- visuailize boundary ---*/
    for (Loop l: hmesh->loops) {
        std::vector<glm::vec3> nodes;
        std::vector<double> value;
        std::vector<std::array<size_t, 2>> edges;
        size_t n = 0;
        for (Half h: l.adjHalfs()) {
            Row3d p1 = h.tail().pos();
            Row3d p2 = h.head().pos();
            nodes.emplace_back(p1.x(), p1.y(), p1.z());
            nodes.emplace_back(p2.x(), p2.y(), p2.z());
            edges.emplace_back(std::array{ n, n + 1 });
            value.emplace_back(n);
            n += 2;
        }
        auto pn = polyscope::registerCurveNetwork("boundaryloop_" + std::to_string(l.id), nodes, edges);
        pn->addEdgeScalarQuantity("value", value);
        pn->resetTransform();
        pn->setRadius(0.001);
    }

    /*--- visuailize generators ---*/
    {
        std::vector<glm::vec3> nodes;
        std::vector<double> value;
        std::vector<std::array<size_t, 2>> edges;
        size_t n = 0;
        for (Half h: hmesh->halfs) {
            int val = hgen->generators[h.id];
            if (val > 0) {
                Row3d p1 = h.face().center();
                Row3d p2 = h.twin().face().center();
                nodes.emplace_back(p1.x(), p1.y(), p1.z());
                nodes.emplace_back(p2.x(), p2.y(), p2.z());
                edges.emplace_back(std::array{ n, n + 1 });
                value.emplace_back(val);
                n += 2;
            }
        }
        auto pn = polyscope::registerCurveNetwork("generator", nodes, edges);
        pn->addEdgeScalarQuantity("value", value);
        pn->resetTransform();
        pn->setRadius(0.001);
    }

    polyscope::show();
}
