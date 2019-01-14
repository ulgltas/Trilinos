#include "Stokhos_my_gemv_MP_Vector_decl.hpp"

template<class Storage,
         class VA,
         class VX,
         class VY>
void my_gemv (const char trans[],
      typename VA::const_value_type& alpha,
      const VA& A,
      const VX& x,
      typename VY::const_value_type& beta,
      const VY& y)
{
  // y := alpha*A*x + beta*y,

  static_assert (Kokkos::View<DA,PA...>::rank == 2, "GEMM: A must have rank 2 (be a matrix).");
  static_assert (Kokkos::View<DX,PX...>::rank == 1, "GEMM: x must have rank 1 (be a vector).");
  static_assert (Kokkos::View<DY,PY...>::rank == 1, "GEMM: y must have rank 1 (be a vector).");
  
  // Get the dimensions
  const size_t m = y.dimension_0 ();
  const size_t n = x.dimension_0 ();
  
  //static_assert( false, "Error: building gemv" );
}
