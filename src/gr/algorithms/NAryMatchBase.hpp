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

#include "gr/algorithms/NAryMatchBase.h"
#include "gr/shared.h"
#include "gr/sampling.h"
#include "gr/utils/logger.h"

#ifdef TEST_GLOBAL_TIMINGS
#   include "../utils/timer.h"
#endif


namespace gr {

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
NAryMatchBase<PointType, TransformVisitor, OptExts ... >
::NAryMatchBase(const typename NAryMatchBase<PointType, TransformVisitor, OptExts ... >
::OptionsType &options,
                      const Utils::Logger& logger
                       )
    : MatchBaseType(options, logger)
{}

template <typename PointType, typename TransformVisitor, template < class, class > typename ... OptExts>
NAryMatchBase<PointType, TransformVisitor, OptExts ... >
::~NAryMatchBase(){}


} // namespace gr
