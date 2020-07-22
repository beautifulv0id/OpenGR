
#include <vector>
#include <atomic>
#include <chrono>
#include <numeric> // std::iota
#include <Eigen/Dense>
#include <sdpa_call.h>


#ifdef OpenGR_USE_OPENMP
#include <omp.h>
#endif

#include "gr/algorithms/GRET_SDP.h"
#include "gr/shared.h"
#include "gr/sampling.h"
#include "gr/utils/logger.h"
#include "gr/accelerators/SDPAWrapper.h"


#ifdef TEST_GLOBAL_TIMINGS
#   include "../utils/timer.h"
#endif


namespace gr {

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
GRET_SDP<PointType, TransformVisitor, OptExts ... >
::GRET_SDP(const typename GRET_SDP<PointType, TransformVisitor, OptExts ... >
::OptionsType &options, const Utils::Logger& logger)
    : MatchBaseType(options, logger)
{}

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
GRET_SDP<PointType, TransformVisitor, OptExts ... >
::~GRET_SDP(){}

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
template<typename PatchRange>
void GRET_SDP<PointType, TransformVisitor, OptExts ... >::RegisterPatches(const PatchRange& patches, const int num_points, TransformVisitor& v){
    n = num_points;
    d = PointType::dim();
    m = patches.size();

    MatrixX L(n+m, n+m);
    MatrixX D(m*d, m*d);
    MatrixX C(m*d, m*d);
    MatrixX B(m*d, n+m);
    Linv = MatrixX(n+m, n+m);

    // Compute matrices B, D and C
    for(int i = 0; i < patches.size(); i++){
        for(const std::pair<PointType,int> &point_with_index : patches[i]){
            int k = point_with_index.second;
            VectorType pos = point_with_index.first.pos();

            L(k,k)++;
            L(k,n+i)--;
            L(n+i,k)--;
            L(n+i,n+i)++;

            B.block(i*d,k,d,1) += pos;
            B.block(i*d,n+i,d,1) -= pos;

            D.block(i*d, i*d, d, d) += pos*pos.transpose();
        }
    }

    // compute Linv, the Moore-Penrose pseudoinverse of L
    Linv = L.completeOrthogonalDecomposition().pseudoInverse();

    // compute C
    C = D - B * Linv * B.transpose();

    // solve the SDP (P2) using C
    MatrixX G(m*d, m*d);

    SDPA_WRAPPER<Scalar> sdpa_solver;
    sdpa_solver.Solve(C,G,d,m);
    //SolveSDP(C, G);

    // compute top d eigenvalues and eigenvectors
    Eigen::EigenSolver<MatrixX> s(G);
    Eigen::VectorXcd eigvals = s.eigenvalues();
    Eigen::MatrixXcd eigvecs = s.eigenvectors();
    std::vector<double> re_eigvals;
    std::vector<Eigen::VectorXd> re_eigvecs;

    for(int i = 0; i < eigvals.rows(); i++) {
        if(eigvals(i).imag() == 0){
            re_eigvals.push_back(eigvals(i).real());
            re_eigvecs.push_back(eigvecs.col(i).real());
        }
    }   

  std::vector<std::pair<double, Eigen::VectorXd>> eig_pairs;
  eig_pairs.reserve(d);
  std::transform(re_eigvals.begin(), re_eigvals.end(), re_eigvecs.begin(), std::back_inserter(eig_pairs),
               [](double a, const Eigen::VectorXd& b) { return std::make_pair(a, b); });

  sort(eig_pairs.begin(), eig_pairs.end(),
    [&](std::pair<double, Eigen::VectorXd>& a, std::pair<double, Eigen::VectorXd>& b) {
        return (a.first > b.first);
    }
  );

  // construct W
  MatrixX W(d, m*d);
  for(int i = 0; i < d; i++){
      W.row(i) = std::sqrt(eig_pairs[i].first) * eig_pairs[i].second.transpose();
  }

  // compute transformations O
  O = MatrixX(d, m*d);
  for(int i = 0; i < m; i++){
      Eigen::Ref<MatrixX> w(W.block(0, i*d, d, d));
      Eigen::JacobiSVD<MatrixX> svd(w, Eigen::ComputeFullU | Eigen::ComputeFullV);
      O.block(0, i*d, d, d) = svd.matrixU() * svd.matrixV().transpose();
  }

  OB = O*B;
}

// returns registered points
template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
template<typename PointRange>
void GRET_SDP<PointType, TransformVisitor, OptExts ... >::getRegisteredPatches(PointRange& registered_points){
  MatrixX R(d, n);
  R = OB*Linv.block(0,0,n+m,n);
  registered_points.reserve(n);
  for(int k = 0; k < n; k++){
    registered_points.push_back(PointType(VectorType(R.block(0,k,d,1))));
  }
}

// returns transformations
template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
template<typename TrRange>
void GRET_SDP<PointType, TransformVisitor, OptExts ... >::getTransformations(TrRange& transformations){
  MatrixX T(d, m);
  T = OB*Linv.block(0, n, n+m, m);

  // store transformations
  MatrixType trafo;
  transformations.reserve(m);
  for(int i = 0; i < m; i++){
    trafo = MatrixType::Zero();
    trafo.block(0, 0, d, d) = O.block(0, i*d, d, d);
    trafo.block(0, d, d, 1) = T.block(0, i, d, 1);
    trafo(d,d) = 1;
    transformations.emplace_back(trafo);
  }
}

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
void GRET_SDP<PointType, TransformVisitor, OptExts ... >::SolveSDP(Eigen::Ref<const MatrixX> C, Eigen::Ref<MatrixX> G){
  SDPA	Problem;

  // All parameteres are renewed
  Problem.setParameterType(SDPA::PARAMETER_STABLE_BUT_SLOW);

  int mDIM   = d*(d+1)/2*m;
  int nBlock = 1;
  Problem.inputConstraintNumber(mDIM);
  Problem.inputBlockNumber(nBlock);
  Problem.inputBlockSize(1,d*m);
  Problem.inputBlockType(1,SDPA::SDP);

  Problem.initializeUpperTriangleSpace();

  // c vec
  int cnt = 1;
  for(int i = 0; i < m; i++){
    for(int j = 0; j < d; j++){
      Problem.inputCVec(cnt++,1);
      for(int k = j+1; k < d; k++)
        Problem.inputCVec(cnt++,0);
    }
  } 

  // F0 = -C
  for(int i = 0; i < m*d; i++)
    for(int j = i; j < m*d; j++)
      Problem.inputElement(0, 1, i+1, j+1, -C(i,j));

 // Fi
  cnt = 1;
  for(int k = 0; k < m; k++){
    for(int i = 0; i < d; i++){
      for(int j = i; j < d; j++){
        Problem.inputElement(cnt++, 1, k*d+i+1, k*d+j+1, 1);
      }
    }
  }

  Problem.initializeUpperTriangle();
  Problem.initializeSolve();
  Problem.solve();


  double* yMat = Problem.getResultYMat(1);
  Eigen::Map<Eigen::MatrixXd> Gmap(yMat, d*m, d*m);
  G = Gmap;
}


} // namespace gr
