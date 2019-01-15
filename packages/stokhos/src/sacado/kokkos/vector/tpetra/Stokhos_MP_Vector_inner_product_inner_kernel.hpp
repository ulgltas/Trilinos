#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
template<> KOKKOS_INLINE_FUNCTION
void inner_product_kernel<Sacado::MP::Vector<Storage>>(
      Sacado::MP::Vector<Storage> * A,
      Sacado::MP::Vector<Storage> * b,
      Sacado::MP::Vector<Storage> * c)
{
  c[0] += b[0]*A[0];
}
#else
#if n_vectors==1
template<> KOKKOS_INLINE_FUNCTION
void inner_product_kernel<Sacado::MP::Vector<Storage>>(
      Sacado::MP::Vector<Storage> * A,
      Sacado::MP::Vector<Storage> * b,
      Sacado::MP::Vector<Storage> * c)
{
  __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[0],0);
  __m512d b_m_0 = M512D_ENSEMBLE_LOAD(b[0],0);
  c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],0),b_m_0,c_m_0);
  M512D_ENSEMBLE_STORE(c[0],0,c_m_0);
}
#endif

#if n_vectors==2
template<> KOKKOS_INLINE_FUNCTION
void inner_product_kernel<Sacado::MP::Vector<Storage>>(
      Sacado::MP::Vector<Storage> * A,
      Sacado::MP::Vector<Storage> * b,
      Sacado::MP::Vector<Storage> * c)
{
  __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[0],0);
  __m512d c_m_1 = M512D_ENSEMBLE_LOAD(c[0],1);
  __m512d b_m_0 = M512D_ENSEMBLE_LOAD(b[0],0);
  __m512d b_m_1 = M512D_ENSEMBLE_LOAD(b[0],1);
  c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],0),b_m_0,c_m_0);
  c_m_1 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],1),b_m_1,c_m_1);
  M512D_ENSEMBLE_STORE(c[0],0,c_m_0);
  M512D_ENSEMBLE_STORE(c[0],1,c_m_1);
}
#endif

#if n_vectors==3
template<> KOKKOS_INLINE_FUNCTION
void inner_product_kernel<Sacado::MP::Vector<Storage>>(
      Sacado::MP::Vector<Storage> * A,
      Sacado::MP::Vector<Storage> * b,
      Sacado::MP::Vector<Storage> * c)
{
  __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[0],0);
  __m512d c_m_1 = M512D_ENSEMBLE_LOAD(c[0],1);
  __m512d c_m_2 = M512D_ENSEMBLE_LOAD(c[0],2);
  __m512d b_m_0 = M512D_ENSEMBLE_LOAD(b[0],0);
  __m512d b_m_1 = M512D_ENSEMBLE_LOAD(b[0],1);
  __m512d b_m_2 = M512D_ENSEMBLE_LOAD(b[0],2);
  c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],0),b_m_0,c_m_0);
  c_m_1 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],1),b_m_1,c_m_1);
  c_m_2 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],2),b_m_2,c_m_2);
  M512D_ENSEMBLE_STORE(c[0],0,c_m_0);
  M512D_ENSEMBLE_STORE(c[0],1,c_m_1);
  M512D_ENSEMBLE_STORE(c[0],2,c_m_2);
}
#endif

#if n_vectors==4
template<> KOKKOS_INLINE_FUNCTION
void inner_product_kernel<Sacado::MP::Vector<Storage>>(
      Sacado::MP::Vector<Storage> * A,
      Sacado::MP::Vector<Storage> * b,
      Sacado::MP::Vector<Storage> * c)
{
  __m512d c_m_0 = M512D_ENSEMBLE_LOAD(c[0],0);
  __m512d c_m_1 = M512D_ENSEMBLE_LOAD(c[0],1);
  __m512d c_m_2 = M512D_ENSEMBLE_LOAD(c[0],2);
  __m512d c_m_3 = M512D_ENSEMBLE_LOAD(c[0],3);
  __m512d b_m_0 = M512D_ENSEMBLE_LOAD(b[0],0);
  __m512d b_m_1 = M512D_ENSEMBLE_LOAD(b[0],1);
  __m512d b_m_2 = M512D_ENSEMBLE_LOAD(b[0],2);
  __m512d b_m_3 = M512D_ENSEMBLE_LOAD(b[0],3);
  c_m_0 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],0),b_m_0,c_m_0);
  c_m_1 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],1),b_m_1,c_m_1);
  c_m_2 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],2),b_m_2,c_m_2);
  c_m_3 = _mm512_fmadd_pd(M512D_ENSEMBLE_LOAD(A[0],3),b_m_3,c_m_3);
  M512D_ENSEMBLE_STORE(c[0],0,c_m_0);
  M512D_ENSEMBLE_STORE(c[0],1,c_m_1);
  M512D_ENSEMBLE_STORE(c[0],2,c_m_2);
  M512D_ENSEMBLE_STORE(c[0],3,c_m_3);
}
#endif
#endif
