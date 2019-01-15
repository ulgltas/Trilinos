#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
template<> KOKKOS_INLINE_FUNCTION
void update_kernel<Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>>>(
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * A,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> alpha,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * b,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * c,
      int i_max)
{
  Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> alphab;
  alphab = alpha*b[0];
  
  for ( int i=0; i<i_max; ++i )
  {
    c[i] += alphab*A[i];
  }
}
#else
#if n_vectors==1
template<> KOKKOS_INLINE_FUNCTION
void update_kernel<Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>>>(
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * A,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> alpha,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * b,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * c,
      int i_max)
{
  __m512d alphab_0 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],0),M512D_ENSEMBLE_LOAD(alpha,0));
  
  for ( int i=0; i<i_max; ++i )
  {
    __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[i],0);
    c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],0),alphab_0,c_m_0);
    M512D_ENSEMBLE_STORE(c[i],0,c_m_0);
  }
}
#endif

#if n_vectors==2
template<> KOKKOS_INLINE_FUNCTION
void update_fused_inner_kernel_II_1<Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>>>(
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * A,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> alpha,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * b,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * c,
      int i_max)
{
  __m512d alphab_0 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],0),M512D_ENSEMBLE_LOAD(alpha,0));
  __m512d alphab_1 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],1),M512D_ENSEMBLE_LOAD(alpha,1));
  
  for ( int i=0; i<i_max; ++i )
  {
    __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[i],0);
    __m512d c_m_1 = M512D_ENSEMBLE_LOAD(c[i],1);
    c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],0),alphab_0,c_m_0);
    c_m_1 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],1),alphab_1,c_m_1);
    M512D_ENSEMBLE_STORE(c[i],0,c_m_0);
    M512D_ENSEMBLE_STORE(c[i],1,c_m_1);
  }
}
#endif

#if n_vectors==3
template<> KOKKOS_INLINE_FUNCTION
void update_fused_inner_kernel_II_1<Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>>>(
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * A,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> alpha,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * b,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * c,
      int i_max)
{
  __m512d alphab_0 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],0),M512D_ENSEMBLE_LOAD(alpha,0));
  __m512d alphab_1 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],1),M512D_ENSEMBLE_LOAD(alpha,1));
  __m512d alphab_2 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],2),M512D_ENSEMBLE_LOAD(alpha,2));
  
  for ( int i=0; i<i_max; ++i )
  {
    __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[i],0);
    __m512d c_m_1 = M512D_ENSEMBLE_LOAD(c[i],1);
    __m512d c_m_2 = M512D_ENSEMBLE_LOAD(c[i],2);
    c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],0),alphab_0,c_m_0);
    c_m_1 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],1),alphab_1,c_m_1);
    c_m_2 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],2),alphab_2,c_m_2);
    M512D_ENSEMBLE_STORE(c[i],0,c_m_0);
    M512D_ENSEMBLE_STORE(c[i],1,c_m_1);
    M512D_ENSEMBLE_STORE(c[i],2,c_m_2);
  }
}
#endif

#if n_vectors==4
template<> KOKKOS_INLINE_FUNCTION
void update_fused_inner_kernel_II_1<Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>>>(
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * A,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> alpha,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * b,
      Sacado::MP::Vector<Stokhos::StaticFixedStorage<int,double,size,Kokkos::OpenMP>> * c,
      int i_max)
{
  __m512d alphab_0 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],0),M512D_ENSEMBLE_LOAD(alpha,0));
  __m512d alphab_1 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],1),M512D_ENSEMBLE_LOAD(alpha,1));
  __m512d alphab_2 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],2),M512D_ENSEMBLE_LOAD(alpha,2));
  __m512d alphab_3 = _mm512_mul_pd(M512D_ENSEMBLE_LOAD(b[0],3),M512D_ENSEMBLE_LOAD(alpha,3));
  
  for ( int i=0; i<i_max; ++i )
  {
    __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[i],0);
    __m512d c_m_1 = M512D_ENSEMBLE_LOAD(c[i],1);
    __m512d c_m_2 = M512D_ENSEMBLE_LOAD(c[i],2);
    __m512d c_m_3 = M512D_ENSEMBLE_LOAD(c[i],3);
    c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],0),alphab_0,c_m_0);
    c_m_1 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],1),alphab_1,c_m_1);
    c_m_2 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],2),alphab_2,c_m_2);
    c_m_3 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[i],3),alphab_3,c_m_3);
    M512D_ENSEMBLE_STORE(c[i],0,c_m_0);
    M512D_ENSEMBLE_STORE(c[i],1,c_m_1);
    M512D_ENSEMBLE_STORE(c[i],2,c_m_2);
    M512D_ENSEMBLE_STORE(c[i],3,c_m_3);
  }
}
#endif
#endif
