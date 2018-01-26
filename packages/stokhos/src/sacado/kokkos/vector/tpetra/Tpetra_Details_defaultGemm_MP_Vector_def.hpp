#include "Tpetra_Details_defaultGemm_MP_Vector.hpp"


#include <iostream>

namespace Tpetra {
namespace Details {
namespace Blas {
namespace Default {

template<class ViewType1,
         class ViewType2,
         class ViewType3,
         class Storage,
         class IndexType>
void
gemm (const char transA,
      const char transB,
      const Sacado::MP::Vector<Storage>& alpha,
      const ViewType1& A,
      const ViewType2& B,
      const Sacado::MP::Vector<Storage>& beta,
      const ViewType3& C)
{

  typedef Sacado::MP::Vector<Storage> Scalar;
  typedef Kokkos::View<Scalar***, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>> ScalarViewType;

  // Assert that A, B, and C are in fact matrices
  static_assert (ViewType1::rank == 2, "GEMM: A must have rank 2 (be a matrix).");
  static_assert (ViewType2::rank == 2, "GEMM: B must have rank 2 (be a matrix).");
  static_assert (ViewType3::rank == 2, "GEMM: C must have rank 2 (be a matrix).");

  // Get the dimensions
  const IndexType m = C.dimension_0 ();
  const IndexType n = C.dimension_1 ();
  const IndexType k = (transA == 'N' || transA == 'n') ?
    A.dimension_1 () : A.dimension_0 ();

  using KokkosBatched::Experimental::Trans;
  using KokkosBatched::Experimental::Algo;

  const int pool_size = Kokkos::DefaultExecutionSpace::thread_pool_size();

  if ( (m < pool_size) && (k < pool_size))
  {
    // Do not use parallel loop as the matrices are too small
    if(transA == 'N' || transA == 'n')
    {
      if (transB == 'N' || transB == 'n')
          KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Blocked>
          ::invoke(alpha, A, B, beta, C);
      else
          KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::Transpose,Algo::Gemm::Blocked>
          ::invoke(alpha, A, B, beta, C);
    }
    else
    {
      if (transB == 'N' || transB == 'n')
          KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::NoTranspose,Algo::Gemm::Blocked>
          ::invoke(alpha, A, B, beta, C);
      else
          KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::Transpose,Algo::Gemm::Blocked>
          ::invoke(alpha, A, B, beta, C);
    } 
  }
  else if(m < k)
  {
    //Inner product: 
    // We split the k columnss of A (if transA == 'N') or the k rows of A (else) in 
    // pool_size groups of at ceil(k/pool_size) (or possibly ceil(k/pool_size)-1 
    // for the last one).
    // We do the same for the k rows of B (if transB == 'N') or the k columns  of B (else).
    const IndexType k_per_thread = ceil(((double ) k)/pool_size);

    for (IndexType i = 0; i < m; ++i) {       
      for (IndexType j = 0; j < n; ++j) {     
        C(i,j) = beta*C(i,j);     
      }
    }

    ScalarViewType C_threads("C_threads", pool_size,m,n);

    Kokkos::deep_copy(C_threads,0);

    if (transA == 'N' || transA == 'n')
    {
      if (transB == 'N' || transB == 'n')
      {
        /*
        Kokkos::parallel_reduce (pool_size, KOKKOS_LAMBDA (const int i, ViewType3 &lC) {
          IndexType k_max = ((i+1)*k_per_thread > k)? k:(i+1)*k_per_thread ;
          auto A_sub = subview (A, Kokkos::ALL(), Kokkos::make_pair (i*k_per_thread,  k_max));
          auto B_sub = subview (B, Kokkos::make_pair (i*k_per_thread,  k_max), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B_sub, beta, lC);    
        },C);  
          */      

        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType k_max = ((i+1)*k_per_thread > k)? k:(i+1)*k_per_thread ;
          auto A_sub = subview (A, Kokkos::ALL(), Kokkos::make_pair (i*k_per_thread,  k_max));
          auto B_sub = subview (B, Kokkos::make_pair (i*k_per_thread,  k_max), Kokkos::ALL());
          auto C_sub = subview (C_threads, i, Kokkos::ALL(), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B_sub, beta, C_sub); 

          for (IndexType i = 0; i < m; ++i) {       
            for (IndexType j = 0; j < n; ++j) {     
              Kokkos::atomic_add(&C(i,j),C_sub(i,j));     
            }
          }
        });             
      }
      else
      {
        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType k_max = ((i+1)*k_per_thread > k)? k:(i+1)*k_per_thread ;
          auto A_sub = subview (A, Kokkos::ALL(), Kokkos::make_pair (i*k_per_thread,  k_max));
          auto B_sub = subview (B, Kokkos::ALL(), Kokkos::make_pair (i*k_per_thread,  k_max));
          auto C_sub = subview (C_threads, i, Kokkos::ALL(), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::Transpose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B_sub, beta, C_sub); 

          for (IndexType i = 0; i < m; ++i) {       
            for (IndexType j = 0; j < n; ++j) {     
              Kokkos::atomic_add(&C(i,j),C_sub(i,j));     
            }
          }
        });                   
      }
    }  
    else
    {
      if (transB == 'N' || transB == 'n')
      {      
        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType k_max = ((i+1)*k_per_thread > k)? k:(i+1)*k_per_thread ;
          auto A_sub = subview (A, Kokkos::make_pair (i*k_per_thread,  k_max), Kokkos::ALL());
          auto B_sub = subview (B, Kokkos::make_pair (i*k_per_thread,  k_max), Kokkos::ALL());
          auto C_sub = subview (C_threads, i, Kokkos::ALL(), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::NoTranspose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B_sub, beta, C_sub); 

          for (IndexType i = 0; i < m; ++i) {       
            for (IndexType j = 0; j < n; ++j) {     
              Kokkos::atomic_add(&C(i,j),C_sub(i,j));     
            }
          }
        });           
      }
      else
      {
        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType k_max = ((i+1)*k_per_thread > k)? k:(i+1)*k_per_thread ;
          auto A_sub = subview (A, Kokkos::make_pair (i*k_per_thread,  k_max), Kokkos::ALL());
          auto B_sub = subview (B, Kokkos::ALL(), Kokkos::make_pair (i*k_per_thread,  k_max));
          auto C_sub = subview (C_threads, i, Kokkos::ALL(), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::Transpose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B_sub, beta, C_sub); 

          for (IndexType i = 0; i < m; ++i) {       
            for (IndexType j = 0; j < n; ++j) {     
              Kokkos::atomic_add(&C(i,j),C_sub(i,j));     
            }
          }
        });         
      }
    }        
  }
  else
  {
    //Update: 
    // We split the m rows of A (if transA == 'N') or the m columns of A (else) in 
    // pool_size groups of at ceil(m/pool_size) (or possibly ceil(m/pool_size)-1 
    // for the last one).
    // We do the same for the rows of C.
    const IndexType m_per_thread = ceil(((double ) m)/pool_size);


    if (transA == 'N' || transA == 'n')
    {
      if (transB == 'N' || transB == 'n')
      {
        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType m_max = ((i+1)*m_per_thread > m)? m:(i+1)*m_per_thread ;
          auto A_sub = subview (A, Kokkos::make_pair (i*m_per_thread,  m_max), Kokkos::ALL());
          auto C_sub = subview (C, Kokkos::make_pair (i*m_per_thread,  m_max), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B, beta, C_sub);    
        });
      }
      else
      {
        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType m_max = ((i+1)*m_per_thread > m)? m:(i+1)*m_per_thread ;
          auto A_sub = subview (A, Kokkos::make_pair (i*m_per_thread,  m_max), Kokkos::ALL());
          auto C_sub = subview (C, Kokkos::make_pair (i*m_per_thread,  m_max), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::Transpose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B, beta, C_sub);    
        });
      }
    }
    else   
    {
      if (transB == 'N' || transB == 'n')
      {
        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType m_max = ((i+1)*m_per_thread > m)? m:(i+1)*m_per_thread ;
          auto A_sub = subview (A, Kokkos::ALL(), Kokkos::make_pair (i*m_per_thread,  m_max));
          auto C_sub = subview (C, Kokkos::make_pair (i*m_per_thread,  m_max), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::NoTranspose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B, beta, C_sub);    
        });
      }
      else
      {
        Kokkos::parallel_for (pool_size, KOKKOS_LAMBDA (const int i) {
          IndexType m_max = ((i+1)*m_per_thread > m)? m:(i+1)*m_per_thread ;
          auto A_sub = subview (A, Kokkos::ALL(), Kokkos::make_pair (i*m_per_thread,  m_max));
          auto C_sub = subview (C, Kokkos::make_pair (i*m_per_thread,  m_max), Kokkos::ALL());
          KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::Transpose,Algo::Gemm::Blocked>
            ::invoke(alpha, A_sub, B, beta, C_sub);    
        });
      }
    }       
  }


  /*

  using KokkosBatched::Experimental::Trans;
  using KokkosBatched::Experimental::Algo;

  if(transA == 'N' || transA == 'n')
  {
    if (transB == 'N' || transB == 'n')
        KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Blocked>
        ::invoke(alpha, A, B, beta, C);
    else
        KokkosBatched::Experimental::SerialGemm<Trans::NoTranspose,Trans::Transpose,Algo::Gemm::Blocked>
        ::invoke(alpha, A, B, beta, C);
  }
  else
  {
    if (transB == 'N' || transB == 'n')
        KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::NoTranspose,Algo::Gemm::Blocked>
        ::invoke(alpha, A, B, beta, C);
    else
        KokkosBatched::Experimental::SerialGemm<Trans::Transpose,Trans::Transpose,Algo::Gemm::Blocked>
        ::invoke(alpha, A, B, beta, C);
  }
  */  
}

} // namespace Default
} // namespace Blas
} // namespace Details
} // namespace Tpetra

#define TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR_ETI(STORAGE) \
typedef Sacado::MP::Vector<STORAGE> SCALAR;\
typedef Kokkos::View<SCALAR**, Kokkos::LayoutRight, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>> VT; \
template void gemm<VT,VT,VT,SCALAR,int> (const char transA, \
      const char transB, \
      const SCALAR& alpha, \
      const VT& A, \
      const VT& B, \
      const SCALAR& beta, \
      const VT& C); \



#define TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR( S, LO, GO, N ) \
typedef Kokkos::View<S**, Kokkos::LayoutRight, N, Kokkos::MemoryTraits<(unsigned int)0>> ViewType; \
  template void \
  Details::Blas::Default::gemm<ViewType, ViewType, ViewType,S,int> \
   (const char,\
    const char, \
    const S&, \
    const ViewType&, \
    const ViewType&, \
    const S&, \
    const ViewType& C); \


#define TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR_STORAGE( STORAGE, LO, GO, N ) \
typedef Kokkos::View<Sacado::MP::Vector<STORAGE>**, Kokkos::LayoutRight, N, Kokkos::MemoryTraits<(unsigned int)0>> ViewType; \
  template void \
  Details::Blas::Default::gemm<ViewType, ViewType, ViewType,STORAGE,int> \
   (const char,\
    const char, \
    const Sacado::MP::Vector<STORAGE>&, \
    const ViewType&, \
    const ViewType&, \
    const Sacado::MP::Vector<STORAGE>&, \
    const ViewType& C); \

/*
#define TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR( ViewType1, ViewType2, ViewType3, Storage, IndexType ) \
  template void \
  Details::Blas::Default::gemm<ViewType1, ViewType2, ViewType3,Storage,IndexType> \
   (const char,\
    const char, \
    const Sacado::MP::Vector<Storage>&, \
    const ViewType1&, \
    const ViewType2&, \
    const Sacado::MP::Vector<Storage>&, \
    const ViewType3& C); \
    */