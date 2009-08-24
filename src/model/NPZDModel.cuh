/**
 * @file
 *
 * @author Generated by spec2x
 * $Rev$
 * $Date$
 */
#ifndef BIM_NPZD_NPZD_CUH
#define BIM_NPZD_NPZD_CUH

#include "BetaPNode.cuh"
#include "BetaZNode.cuh"
#include "BetaNNode.cuh"
#include "BetaDNode.cuh"
#include "TNode.cuh"
#include "ENode.cuh"
#include "LNode.cuh"
#include "BetaENode.cuh"
#include "MuCNode.cuh"
#include "MuCNNode.cuh"
#include "MuChNode.cuh"
#include "NuANode.cuh"
#include "PiNCNode.cuh"
#include "ZetaINode.cuh"
#include "ZetaClNode.cuh"
#include "ZetaENode.cuh"
#include "NuRNode.cuh"
#include "ZetaQNode.cuh"
#include "ZetaLNode.cuh"
#include "PNode.cuh"
#include "ZNode.cuh"
#include "DNode.cuh"
#include "NNode.cuh"
#include "ChlaNode.cuh"
#include "EZNode.cuh"

#include "bi/model/NodeSpec.hpp"
#include "bi/model/BayesNet.hpp"

/**
 * In-net spec.
 */
BEGIN_NODESPEC(NPZDInSpec)
SINGLE_TYPE(1, MuCNode)
SINGLE_TYPE(1, MuCNNode)
SINGLE_TYPE(1, MuChNode)
SINGLE_TYPE(1, NuANode)
SINGLE_TYPE(1, PiNCNode)
SINGLE_TYPE(1, ZetaINode)
SINGLE_TYPE(1, ZetaClNode)
SINGLE_TYPE(1, ZetaENode)
SINGLE_TYPE(1, NuRNode)
SINGLE_TYPE(1, ZetaQNode)
SINGLE_TYPE(1, ZetaLNode)
SINGLE_TYPE(1, ChlaNode)
SINGLE_TYPE(1, EZNode)
END_NODESPEC()

/**
 * Ex-net spec
 */
BEGIN_NODESPEC(NPZDExSpec)
SINGLE_TYPE(1, PNode)
SINGLE_TYPE(1, ZNode)
SINGLE_TYPE(1, DNode)
SINGLE_TYPE(1, NNode)
END_NODESPEC()

/**
 * R-net spec
 */
BEGIN_NODESPEC(NPZDRSpec)

END_NODESPEC()

/**
 * F-net spec
 */
BEGIN_NODESPEC(NPZDFSpec)
SINGLE_TYPE(1, BetaPNode)
SINGLE_TYPE(1, BetaZNode)
SINGLE_TYPE(1, BetaNNode)
SINGLE_TYPE(1, BetaDNode)
SINGLE_TYPE(1, TNode)
SINGLE_TYPE(1, ENode)
SINGLE_TYPE(1, LNode)
SINGLE_TYPE(1, BetaENode)
END_NODESPEC()

/**
 * 
 */
class NPZDModel : public bi::BayesNet<
    GET_NODESPEC(NPZDInSpec),
    GET_NODESPEC(NPZDExSpec),
    GET_NODESPEC(NPZDRSpec),
    GET_NODESPEC(NPZDFSpec)> {
public:
  /**
   * Constructor.
   */
  NPZDModel();

  /*
   * Nodes.
   */
  BetaPNode betaP;
  BetaZNode betaZ;
  BetaNNode betaN;
  BetaDNode betaD;
  TNode T;
  ENode E;
  LNode L;
  BetaENode betaE;
  MuCNode muC;
  MuCNNode muCN;
  MuChNode muCh;
  NuANode nuA;
  PiNCNode piNC;
  ZetaINode zetaI;
  ZetaClNode zetaCl;
  ZetaENode zetaE;
  NuRNode nuR;
  ZetaQNode zetaQ;
  ZetaLNode zetaL;
  PNode P;
  ZNode Z;
  DNode D;
  NNode N;
  ChlaNode Chla;
  EZNode EZ;

};

#endif

