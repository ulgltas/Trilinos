#ifndef KOKKOSBLAS2_GEMV_MP_VECTOR_HPP_
#define KOKKOSBLAS2_GEMV_MP_VECTOR_HPP_

#include <type_traits>
#include "Sacado_ConfigDefs.h"

#include "Stokhos_ViewStorage.hpp"
#include "Sacado_MP_Vector.hpp"
#include "Kokkos_View_MP_Vector.hpp"
#include "Kokkos_ArithTraits_MP_Vector.hpp"
#include "KokkosBlas.hpp"

#include "Stokhos_my_gemv_MP_Vector_decl.hpp"

namespace KokkosBlas {
template<typename DA, typename ... PA,
         typename DX, typename ... PX,
         typename DY, typename ... PY>
typename std::enable_if< Kokkos::is_view_mp_vector< Kokkos::View<DA,PA...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DX,PX...> >::value &&
                         Kokkos::is_view_mp_vector< Kokkos::View<DY,PY...> >::value >::type
gemv (const char trans[],
      typename Kokkos::View<DA,PA...>::const_value_type& alpha,
      const Kokkos::View<DA,PA...>& A,
      const Kokkos::View<DX,PX...>& x,
      typename Kokkos::View<DY,PY...>::const_value_type& beta,
      const Kokkos::View<DY,PY...>& y)
{
  typedef typename Kokkos::View<DA,PA...>::value_type Scalar;
  typedef Kokkos::View<DA,PA...> VA;
  typedef Kokkos::View<DX,PX...> VX;
  typedef Kokkos::View<DY,PY...> VY;
  my_gemv<Scalar,VA,VX,VY>(trans,alpha,A,x,beta,y);
}
  
}
#endif
