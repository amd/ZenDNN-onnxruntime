#pragma once

#include "orttraining/training_ops/cuda/bert/fmha/iterators/predicated_tile_access_iterator_residual_last.h"
#include "orttraining/training_ops/cuda/bert/fmha/iterators/predicated_tile_iterator_residual_last.h"

namespace cutlass {
namespace transform {
namespace threadblock {

template <typename BaseIterator>
struct MakeIteratorResidualLast;

template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    int AccessSize,
    bool Gather>
struct MakeIteratorResidualLast<PredicatedTileIterator<
    Shape,
    Element,
    Layout,
    AdvanceRank,
    ThreadMap,
    AccessSize,
    Gather>> {
  using Iterator = PredicatedTileIteratorResidualLast<
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessSize,
      Gather>;
};

template <
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    typename AccessType,
    bool Gather>
struct MakeIteratorResidualLast<PredicatedTileAccessIterator<
    Shape,
    Element,
    Layout,
    AdvanceRank,
    ThreadMap,
    AccessType,
    Gather>> {
  using Iterator = PredicatedTileAccessIteratorResidualLast<
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessType,
      Gather>;
};
} // namespace threadblock
} // namespace transform
} // namespace cutlass
