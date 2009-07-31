/**
 * @file
 * 
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_ZETALNODE_CUH
#define BIM_NPZD_ZETALNODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$\zeta^L\f$; Z linear mortality term
 *
 * \f$\zeta^L(1-R_Z) + R_Z  (\alpha^L + \sigma^L u^L)\f$
 */
class ZetaLNode : public bi::BayesNode {
public:
  template<class V1, class V2, class V3, class V4>
  static CUDA_FUNC_BOTH void s(const V1& fpax, const V2& rpax,
      const V3& inpax, V4& x);
};

#include "bi/model/NodeStaticTraits.hpp"
#include "bi/model/NodeTypeTraits.hpp"

IS_GENERIC_STATIC(ZetaLNode)
IS_IN_NODE(ZetaLNode)

template<class V1, class V2, class V3, class V4>
inline void ZetaLNode::s(const V1& fpax, const V2& rpax,
    const V3& inpax, V4& x) {
  static const real_t RZ = CUDA_REAL(0.0);

  const real_t zetaL = inpax[0];
  const real_t alphaL = inpax[1];
  const real_t sigmaL = inpax[2];
  const real_t uL = rpax[0];

  x = zetaL*(1 - RZ) + RZ*(alphaL + sigmaL*uL);
}

#endif

