/**
 * @file
 * 
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_DNODE_CUH
#define BIM_NPZD_DNODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$D\f$; 
 *
 * \f$(1 - \zeta^E) \delta^I \pi^G + \zeta^M - \nu^R - \delta^S D/L + \beta^E (\beta^D - D)\f$
 */
class DNode : public bi::BayesNode {
public:
  template<class T, class V1, class V2, class V3, class V4, class V5>
  static CUDA_FUNC_BOTH void dfdt(const T t, const V1& fpax,
      const V2& rpax, const V3& inpax, const V4& expax, V5& dfdt);
};

#include "bi/model/NodeForwardTraits.hpp"
#include "bi/model/NodeTypeTraits.hpp"

IS_EX_NODE(DNode)
IS_ODE_FORWARD(DNode)

template<class T1, class V1, class V2, class V3, class V4, class V5>
inline void DNode::dfdt(const T1 t, const V1& fpax,
    const V2& rpax, const V3& inpax, const V4& expax, V5& dfdt) {
  const real_t deltaI = CUDA_REAL(0);
  const real_t tau10 = CUDA_REAL(0);
  const real_t tauR = CUDA_REAL(0);
  const real_t piK = CUDA_REAL(0);
  const real_t deltaS = CUDA_REAL(0);

  const real_t zetaE = inpax[0];
  const real_t Z = expax[0];
  const real_t zetaI = inpax[1];
  const real_t T = fpax[0];
  const real_t zetaCl = inpax[2];
  const real_t P = expax[1];
  const real_t zetaQ = inpax[3];
  const real_t zetaL = inpax[4];
  const real_t nuR = inpax[5];
  const real_t D = expax[2];
  const real_t L = fpax[1];
  const real_t betaE = fpax[2];
  const real_t betaD = fpax[3];

  const real_t tauC = pow(tau10, (T - tauR)/10);
  const real_t piS = pow(zetaCl*P/zetaI, piK);
  const real_t piG = Z*zetaI*tauC*piS/(1+piS);
  const real_t zetaM = (zetaQ*Z + zetaL)*Z;

  dfdt = (1 - zetaE)*deltaI*piG + zetaM - nuR - deltaS*D/L + betaE*(betaD - D);
}

#endif
