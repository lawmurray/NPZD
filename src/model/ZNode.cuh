/**
 * @file
 *
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_ZNODE_CUH
#define BIM_NPZD_ZNODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$Z\f$; 
 *
 * \f$\pi^G \zeta^E - \zeta^M + \beta^E(\beta^Z - Z)\f$
 */
class ZNode : public bi::BayesNode {
public:
  /**
   * Constructor.
   */
  ZNode();

  template<class T, class V1, class V2, class V3, class V4, class V5>
  static CUDA_FUNC_BOTH void dfdt(const T t, const V1& fpax,
      const V2& rpax, const V3& inpax, const V4& expax, V5& dfdt);
};

#include "bi/model/NodeForwardTraits.hpp"
#include "bi/model/NodeTypeTraits.hpp"

IS_EX_NODE(ZNode)
IS_ODE_FORWARD(ZNode)

inline ZNode::ZNode() {
  setName("Z");
}

template<class T1, class V1, class V2, class V3, class V4, class V5>
inline void ZNode::dfdt(const T1 t, const V1& fpax,
    const V2& rpax, const V3& inpax, const V4& expax, V5& dfdt) {
  const real_t tau10 = CUDA_REAL(2);
  const real_t tauR = CUDA_REAL(15);
  const real_t piK = CUDA_REAL(2);

  const real_t Z = expax[0];
  const real_t zetaI = inpax[0];
  const real_t T = fpax[0];
  const real_t zetaCl = inpax[1];
  const real_t P = expax[1];
  const real_t zetaE = inpax[2];
  const real_t zetaQ = inpax[3];
  const real_t zetaL = inpax[4];
  const real_t betaE = fpax[1];
  const real_t betaZ = fpax[2];

  const real_t tauC = pow(tau10, (T - tauR)/10);
  const real_t piS = pow(zetaCl*P/zetaI, piK);
  const real_t piG = Z*zetaI*tauC*piS/(1+piS);
  const real_t zetaM = (zetaQ*Z + zetaL)*Z;

  dfdt = piG*zetaE - zetaM + betaE*(betaZ - Z);
}

#endif

