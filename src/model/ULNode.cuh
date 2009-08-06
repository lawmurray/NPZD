/**
 * @file
 * 
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_ULNODE_CUH
#define BIM_NPZD_ULNODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$u^L\f$; 
 *
 * \f$u^L \sim \mathcal{N}(0,1)\f$
 */
class ULNode : public bi::BayesNode {
public:

};

#include "bi/model/NodeRandomTraits.hpp"
#include "bi/model/NodeTypeTraits.hpp"

IS_GAUSSIAN(ULNode)
IS_R_NODE(ULNode)

#endif
