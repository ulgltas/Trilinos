#ifndef KOKKOSBLAS2_GEMV_MP_VECTOR_HPP_
#define KOKKOSBLAS2_GEMV_MP_VECTOR_HPP_


#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"
#include "Kokkos_Core.hpp"

#include "KokkosBatched_Vector.hpp"


#include "KokkosBatched_Gemm_Decl.hpp"

#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include <iostream>



template<class DA, class ... PA,
         class DX, class ... PX,
         class DY, class ... PY>
typename std::enable_if< Kokkos::is_view_mp_vector< Kokkos::View<DA,PA...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DB,PB...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DC,PC...> >::value >::type
gemv (const char trans[],
      typename Kokkos::View<DA,PA...>::const_value_type& alpha,
      const Kokkos::View<DA,PA...>& A,
      const Kokkos::View<DX,PX...>& x,
      typename Kokkos::View<DY,PY...>::const_value_type& beta,
      const Kokkos::View<DY,PY...>& y)
{
  // y := alpha*A*x + beta*y,

  // Get the dimensions
  const IndexType m = y.dimension_0 ();
  const IndexType n = x.dimension_0 ();
  

  typedef Sacado::MP::Vector<Storage> Scalar;
  typedef Kokkos::View<Scalar***, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>> ScalarViewType;
  typedef Kokkos::View<IndexType*, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>> IndexViewType;

  using KokkosBatched::Experimental::Trans;
  using KokkosBatched::Experimental::Algo;

  const int pool_size = Kokkos::DefaultExecutionSpace::thread_pool_size();

  if((trans[0] == 'N') || (trans[0] == 'n'))
  {
    KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Blocked>
        ::invoke(alpha, A, x, beta, y);
  }
  else
  {
    KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::NoTranspose,Algo::Gemm::Blocked>
        ::invoke(alpha, A, x, beta, y);
  }
}
#endif
