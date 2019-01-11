#ifndef KOKKOSBLAS2_GEMV_MP_VECTOR_HPP_
#define KOKKOSBLAS2_GEMV_MP_VECTOR_HPP_

template<class DA, class ... PA,
         class DX, class ... PX,
         class DY, class ... PY>
typename std::enable_if< Kokkos::is_view_mp_vector< Kokkos::View<DA,PA...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DX,PX...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DY,PY...> >::value >::type
KokkosBlas::gemv (const char trans[],
      typename Kokkos::View<DA,PA...>::const_value_type& alpha,
      const Kokkos::View<DA,PA...>& A,
      const Kokkos::View<DX,PX...>& x,
      typename Kokkos::View<DY,PY...>::const_value_type& beta,
      const Kokkos::View<DY,PY...>& y)
{
  // y := alpha*A*x + beta*y,

  static_assert (Kokkos::View<DA,PA...>::rank == 2, "GEMM: A must have rank 2 (be a matrix).");
  static_assert (Kokkos::View<DX,PX...>::rank == 1, "GEMM: x must have rank 1 (be a vector).");
  static_assert (Kokkos::View<DY,PY...>::rank == 1, "GEMM: y must have rank 1 (be a vector).");
  
  // Get the dimensions
  const IndexType m = y.dimension_0 ();
  const IndexType n = x.dimension_0 ();
  
}
#endif
