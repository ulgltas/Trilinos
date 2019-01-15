#include "Stokhos_my_gemv_MP_Vector_decl.hpp"
#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"

#ifdef STOKHOS_MP_VECTOR_MASK_USE_II
#include <immintrin.h>
#endif

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

template< typename T>
KOKKOS_INLINE_FUNCTION
void update_kernel(T * A, T alpha, T * b, T * c, int i_max);

template< typename T>
KOKKOS_INLINE_FUNCTION
void inner_product_kernel(T * A, T alpha, T * b, T * c, int i_max);

#define size 4
#define n_vectors 1
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::Serial>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#undef size
#undef n_vectors

#define size 8
#define n_vectors 1
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::Serial>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#undef size
#undef n_vectors

#define size 16
#define n_vectors 2
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::Serial>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#undef size
#undef n_vectors

#define size 24
#define n_vectors 3
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::Serial>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#undef size
#undef n_vectors

#define size 32
#define n_vectors 4
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#define Storage Stokhos::StaticFixedStorage<int,double,size,Kokkos::Serial>
#include "Stokhos_MP_Vector_inner_product_inner_kernel.hpp"
#include "Stokhos_MP_Vector_update_inner_kernel.hpp"
#undef Storage
#undef size
#undef n_vectors

template<class Scalar,
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
  
  const size_t m_c = Sacado_MP_Vector_GEMV_Tile_Size(sizeof(Scalar));
  const size_t n_vectors = Sacado_MP_Vector_GEMV_Number_Vectors(sizeof(Scalar));

  const int n_tiles = ceil(((double) m)/m_c);

  Kokkos::parallel_for (n_tiles, KOKKOS_LAMBDA (const int i_tile)
  {
    int i_min = m_c*i_tile;
    bool last_tile = (i_tile==(n_tiles-1));
    int i_max = (last_tile) ? m : (i_min+m_c);

    #pragma unroll
    for ( size_t i=i_min; i<i_max; ++i )
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
      y(i) = beta*y(i);
#else
      #pragma unroll (n_vectors)
      for ( size_t ell=0; ell<n_vectors; ++ell )
        M512D_ENSEMBLE_STORE(y(i),ell,_mm512_mul_pd(M512D_ENSEMBLE_LOAD<Scalar>(y(i),ell),M512D_ENSEMBLE_LOAD<Scalar>(beta,ell)));
#endif
    }
    
    for ( size_t j=0; j<n; ++j )
      update_kernel<Scalar>(&A(i_min,j),alpha,&x(j),&y(i_min),i_max-i_min);
  });
}

template<class Scalar,
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

  const int pool_size = Kokkos::DefaultExecutionSpace::thread_pool_size();

  const int team_size = 4;

  const size_t m_c = Sacado_MP_Vector_GEMV_Tile_Size(sizeof(Scalar));
  const int n_per_tile2 = m_c*team_size;

  const int n_i2 = ceil(((double) n)/n_per_tile2);
  using Kokkos::TeamThreadRange;

  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>( n_i2, team_size);
  typedef Kokkos::TeamPolicy<>::member_type member_type;
  
  Kokkos::parallel_for (m, KOKKOS_LAMBDA (const int i)
  {
      y(i) *= beta;
  });

  Kokkos::parallel_for (policy, KOKKOS_LAMBDA (member_type team_member)
  {
    const int j = team_member.league_rank();
    const int j_min = n_per_tile2*j;
    const int j_max = (j_min+n_per_tile2 > n) ? n : j_min+n_per_tile2;
    const int i_min = (j<m) ? j:(j%m);

    for ( int i=i_min; i<m; ++i ){
      Scalar tmp = 0.;
      Kokkos::parallel_reduce (TeamThreadRange (team_member, j_max-j_min),[=] (int jj, Scalar& tmp_sum){
        tmp_sum += A(jj+j_min,i) * x(jj+j_min);
      },tmp);
      tmp *= alpha;
      if ( team_member.team_rank () == 0) {
        Kokkos::atomic_add(&y(i),tmp);
      }
    }
  });
}

template<class Scalar,
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
  
  if (trans[0]=='n'||trans[0]=='N')
  {
    my_update<Scalar,VA,VX,VY>(alpha,A,x,beta,y);
  }
  else
  {
    my_inner_product<Scalar,VA,VX,VY>(alpha,A,x,beta,y);
  }
}
