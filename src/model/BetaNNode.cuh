/**
 * @file
 * 
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_BETANNODE_CUH
#define BIM_NPZD_BETANNODE_CUH

#include "bi/model/BayesNode.hpp"
#include "bi/cuda/cuda.hpp"

/**
 * \f$\beta^N\f$; Nitrogen boundary condition
 *
 * \f$\f$
 */
class BetaNNode : public bi::BayesNode {
public:

};

#include "bi/model/NodeTypeTraits.hpp"

IS_F_NODE(BetaNNode)

#endif
