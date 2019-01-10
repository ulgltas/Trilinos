#ifndef KOKKOSBLAS3_GEMM_MP_VECTOR_HPP_
#define KOKKOSBLAS3_GEMM_MP_VECTOR_HPP_

template<class DA, class ... PA,
         class DB, class ... PB,
         class DC, class ... PC>
typename std::enable_if< Kokkos::is_view_mp_vector< Kokkos::View<DA,PA...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DB,PB...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DC,PC...> >::value >::type
gemm (const char transA[],
      const char transB[],
      typename Kokkos::View<DA,PA...>::const_value_type& alpha,
      const Kokkos::View<DA,PA...>& A,
      const Kokkos::View<DB,PB...>& B,
      typename Kokkos::View<DC,PC...>::const_value_type& beta,
      const Kokkos::View<DC,PC...>& C)
{
  // Assert that A, B, and C are in fact matrices
  static_assert (ViewType1::rank == 2, "GEMM: A must have rank 2 (be a matrix).");
  static_assert (ViewType2::rank == 2, "GEMM: B must have rank 2 (be a matrix).");
  static_assert (ViewType3::rank == 2, "GEMM: C must have rank 2 (be a matrix).");
  
  if (C.dimension_1 () == 1)
    gemv();
  else
    throw_error("GEMM");
}

#endif
