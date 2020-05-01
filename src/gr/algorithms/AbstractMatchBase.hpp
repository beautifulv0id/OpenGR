//
// Created by Sandra Alfaro on 24/05/18.
//

#include <vector>
#include <atomic>
#include <chrono>
#include <numeric> // std::iota

#ifdef OpenGR_USE_OPENMP
#include <omp.h>
#endif

#include "gr/algorithms/AbstractMatchBase.h"
#include "gr/shared.h"
#include "gr/sampling.h"
#include "gr/utils/logger.h"

#ifdef TEST_GLOBAL_TIMINGS
#   include "../utils/timer.h"
#endif


#define MATCH_BASE_TYPE AbstractMatchBase<PointType, TransformVisitor, OptExts ... >


namespace gr {

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
MATCH_BASE_TYPE::AbstractMatchBase(const typename MATCH_BASE_TYPE::OptionsType &options,
                      const Utils::Logger& logger
                       )
    : randomGenerator_(options.randomSeed)
    , logger_(logger)
    , options_(options)
{}

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
MATCH_BASE_TYPE::~AbstractMatchBase(){}

} // namespace gr

#undef MATCH_BASE_TYPE
