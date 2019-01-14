#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"
#include "Kokkos_Core.hpp"

template<class Storage,
         class VA,
         class VX,
         class VY>
void my_gemv (const char trans[],
      typename VA::const_value_type& alpha,
      const VA& A,
      const VX& x,
      typename VY::const_value_type& beta,
      const VY& y);
