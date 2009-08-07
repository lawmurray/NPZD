/**
 * @file
 *
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_NNODE_CUH
#define BIM_NPZD_NNODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$N\f$; 
 *
 * \f$-\mu P + (1 - \zeta^E)(1 - \delta^I)\pi^G + \nu^R + \beta^E(\beta^N - N)\f$
 */
class NNode : public bi::BayesNode {
public:
  /**
   * Constructor.
   */
  NNode();

  template<class T, class V1, class V2, class V3, class V4, class V5>
  static CUDA_FUNC_BOTH void dfdt(const T t, const V1& fpax,
      const V2& rpax, const V3& inpax, const V4& expax, V5& dfdt);
};

#include "bi/model/NodeForwardTraits.hpp"
#include "bi/model/NodeTypeTraits.hpp"

IS_EX_NODE(NNode)
IS_ODE_FORWARD(NNode)

inline NNode::NNode() {
  setName("N");
}

template<class T1, class V1, class V2, class V3, class V4, class V5>
inline void NNode::dfdt(const T1 t, const V1& fpax,
    const V2& rpax, const V3& inpax, const V4& expax, V5& dfdt) {
  const real_t tau10 = CUDA_REAL(0);
  const real_t tauR = CUDA_REAL(0);
  const real_t piC = CUDA_REAL(0);
  const real_t piPE = CUDA_REAL(0);
  const real_t deltaI = CUDA_REAL(0);
  const real_t piK = CUDA_REAL(0);

  const real_t muC = inpax[0];
  const real_t T = fpax[0];
  const real_t muCh = inpax[1];
  const real_t E = fpax[1];
  const real_t N = expax[0];
  const real_t muCN = inpax[2];
  const real_t nuA = inpax[3];
  const real_t P = expax[1];
  const real_t zetaE = inpax[4];
  const real_t Z = expax[2];
  const real_t zetaI = inpax[5];
  const real_t zetaCl = inpax[6];
  const real_t nuR = inpax[7];
  const real_t betaE = fpax[2];
  const real_t betaN = fpax[3];

  const real_t tauC = pow(tau10, (T - tauR)/10);
  const real_t muCT = muC*tauC;
  const real_t piQ = piC*piPE;
  const real_t piE = 1-exp(-piQ*muCh*E/muC);
  const real_t piN = muCT/muCN;
  const real_t nuN = N/(N + piN/nuA);
  const real_t mu = muCT*piE*nuN/(nuN + muCN*piE);
  const real_t piS = pow(zetaCl*P/zetaI, piK);
  const real_t piG = Z*zetaI*tauC*piS/(1+piS);

  dfdt = -mu*P + (1 - zetaE)*(1 - deltaI)*piG + nuR + betaE*(betaN - N);
}

#endif

