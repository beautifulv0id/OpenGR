#ifndef _OPENGR_ALGO_GRET_SDP_
#define _OPENGR_ALGO_GRET_SDP_

#include <vector>
#include <Eigen/Dense>

#ifdef OpenGR_USE_OPENMP
#include <omp.h>
#endif

#include "gr/shared.h"
#include "gr/algorithms/NAryMatchBase.h"


#ifdef TEST_GLOBAL_TIMINGS
#   include "gr/utils/timer.h"
#endif

namespace gr{


template < class Derived, class TBase>
class GRET_SDP_Options : public  TBase
{
public:
    using Scalar = typename TBase::Scalar;

};


/// \brief Base class for Congruent Sec Exploration algorithms
/// \tparam _Traits Defines properties of the Base used to build the congruent set.
template <typename _PointType,
          typename _TransformVisitor,
          template < class, class > class ... OptExts >
class GRET_SDP : public NAryMatchBase<_PointType, _TransformVisitor, OptExts ..., GRET_SDP_Options> {

public:
    using TransformVisitor = _TransformVisitor;

    using MatchBaseType = NAryMatchBase<_PointType, _TransformVisitor, OptExts ..., GRET_SDP_Options>;
    using PosMutablePoint = typename MatchBaseType::PosMutablePoint;
    using OptionsType = typename MatchBaseType::OptionsType;

    using Scalar = typename MatchBaseType::Scalar;
    using VectorType = typename MatchBaseType::VectorType;
    using MatrixType = typename MatchBaseType::MatrixType;
    
    using MatrixX = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    using LogLevel = typename MatchBaseType::LogLevel;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    GRET_SDP(const OptionsType& options
            , const Utils::Logger &logger
    );

    virtual ~GRET_SDP();

    // computes registered points and corresponding transformations
    template<typename Solver, typename PatchRange>
    void RegisterPatches(const PatchRange& patches, const int n, TransformVisitor& v);

    // returns registered points
    template<typename PointRange>
    void getRegisteredPatches(PointRange& registered_points);

    // returns transformations
    template<typename TrRange>
    void getTransformations(TrRange& transformations);

    private:
    int d;
    int m;
    int n;

    MatrixX O;
    MatrixX OB;
    MatrixX Linv;
    
    std::vector<std::pair<_PointType, int>> registered_points_;
    std::vector<MatrixType> transformations_;

}; /// class GRET_SDP
} /// namespace gr
#include "GRET_SDP.hpp"

#endif // _OPENGR_ALGO_GRET_SDP_
