/**
 * @file
 *
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_UINODE_CUH
#define BIM_NPZD_UINODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$u^I\f$; 
 *
 * \f$u^I \sim \mathcal{N}(0,1)\f$
 */
class UINode : public bi::BayesNode {
public:
  /**
   * Constructor.
   */
  UINode();

};

#include "bi/model/NodeRandomTraits.hpp"
#include "bi/model/NodeTypeTraits.hpp"

IS_GAUSSIAN(UINode)
IS_R_NODE(UINode)

inline UINode::UINode() {
  setName("uI");
}

#endif

