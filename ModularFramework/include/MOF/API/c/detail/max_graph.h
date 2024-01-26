//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_API_C_DETAIL_MAX_GRAPH_H
#define MOF_API_C_DETAIL_MAX_GRAPH_H

#include "MOF/GraphBuilder/Loc.h"
#include "MOF/GraphBuilder/Module.h"
#include "MOF/GraphBuilder/Symbol.h"
#include "MOF/GraphBuilder/Type.h"

struct M_AttrMap {
  M::MOF::GraphBuilder::AttrMap map;
};

struct M_Graph {
  M::MOF::GraphBuilder::Graph graph;
};

struct M_Tuple {
  M::MOF::GraphBuilder::Tuple tuple;
};

struct M_Symbol {
  M::MOF::GraphBuilder::Symbol symbol;
};

struct M_Arity {
  M::MOF::GraphBuilder::Arity arity;
};

#endif // MOF_API_C_DETAIL_MAX_GRAPH_H
