#include "Stokhos_my_gemv_MP_Vector_decl.hpp"

#define Sacado_MP_Vector_GEMV_Unrolling_Factor(size)                  \
{                                                                     \
  (size==32*8 ? 1 :                                                   \
  (size==24*8 ? 1 :                                                   \
  (size==16*8 ? 1 :                                                   \
  (size==8*8  ? 1 :                                                   \
                1 ))))                                                \
}

#define Sacado_MP_Vector_GEMV_Tile_Size(size)                         \
{                                                                     \
  (size==32*8 ? 16 :                                                  \
  (size==24*8 ? 21 :                                                  \
  (size==16*8 ? 32 :                                                  \
  (size==8*8  ? 64 :                                                  \
                512 ))))                                              \
}

#define Sacado_MP_Vector_GEMV_Number_Vectors(size)                    \
{                                                                     \
  (size==32*8 ? 4 :                                                   \
  (size==24*8 ? 3 :                                                   \
  (size==16*8 ? 2 :                                                   \
  (size==8*8  ? 1 :                                                   \
                1 ))))                                                \
}

#define size 4
#define n_vectors 1
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef size
#undef n_vectors

#define size 8
#define n_vectors 1
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef size
#undef n_vectors

#define size 16
#define n_vectors 2
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef size
#undef n_vectors

#define size 24
#define n_vectors 3
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef size
#undef n_vectors

#define size 32
#define n_vectors 4
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef size
#undef n_vectors

template<class Storage,
         class VA,
         class VX,
         class VY>
void my_update (
      typename VA::const_value_type& alpha,
      const VA& A,
      const VX& x,
      typename VY::const_value_type& beta,
      const VY& y)
{
  // Get the dimensions
  const size_t m = y.dimension_0 ();
  const size_t n = x.dimension_0 ();
  
  typedef Sacado::MP::Vector<Storage> Scalar;
  
  const size_t u   = Sacado_MP_Vector_GEMV_Unrolling_Factor(sizeof(Scalar));
  const size_t m_c = Sacado_MP_Vector_GEMV_Tile_Size(sizeof(Scalar));
  const size_t n_vectors = Sacado_MP_Vector_GEMV_Number_Vectors(sizeof(Scalar));

  const size_t n_tiles = ceil(((double) m)/m_c);

  Kokkos::parallel_for (n_tiles, KOKKOS_LAMBDA (const int i_tile)
  {
    IndexType i_min = m_c*i_tile;
    bool last_tile = (i_tile==(n_tiles-1));
    IndexType i_max = (last_tile) ? m : (i_min+m_c);
    IndexType i_max_u = i_min + (i_max-i_min) - (i_max-i_min)%u;

    #pragma unroll
    for ( IndexType i=i_min; i<i_max; ++i )
      #pragma unroll (n_vectors)
      for ( int ell=0; ell<n_vectors; ++ell )
        M512D_ENSEMBLE_STORE(c(i,0),ell,_mm512_mul_pd(M512D_ENSEMBLE_LOAD<Scalar>(c(i,0),ell),M512D_ENSEMBLE_LOAD<Scalar>(beta,ell)));
    
    for ( IndexType j=0; j<n; ++j )
      update_kernel<Scalar>(&A(i_min,j),alpha,&b(j,0),&c(i_min,0),m,i_max-i_min);
  });
}

template<class Storage,
         class VA,
         class VX,
         class VY>
void my_inner_product (
      typename VA::const_value_type& alpha,
      const VA& A,
      const VX& x,
      typename VY::const_value_type& beta,
      const VY& y)
{
  // Get the dimensions
  const size_t m = y.dimension_0 ();
  const size_t n = x.dimension_0 ();
}

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

  static_assert (VA::rank == 2, "GEMM: A must have rank 2 (be a matrix).");
  static_assert (VX::rank == 1, "GEMM: x must have rank 1 (be a vector).");
  static_assert (VY::rank == 1, "GEMM: y must have rank 1 (be a vector).");
  
  if (trans=='n'||trans=='N')
  {
    my_update<Storage,VA,VX,VY>(alpha,A,x,beta,y);
  }
  else
  {
    my_inner_product<Storage,VA,VX,VY>(alpha,A,x,beta,y);
  }
}
