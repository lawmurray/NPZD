/**
 * @file
 *
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_SIGMAQNODE_CUH
#define BIM_NPZD_SIGMAQNODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$\sigma^Q\f$; 
 *
 * \f$0\f$
 */
class SigmaQNode : public bi::BayesNode {
public:
  /**
   * Constructor.
   */
  SigmaQNode();

};

#include "bi/model/NodeStaticTraits.hpp"
#include "bi/model/NodeTypeTraits.hpp"

IS_IN_NODE(SigmaQNode)

inline SigmaQNode::SigmaQNode() {
  setName("sigmaQ");
}

#endif

