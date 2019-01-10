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
