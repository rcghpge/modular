//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_API_C_DETAIL_MOBI_H
#define MOF_API_C_DETAIL_MOBI_H

#include "MOF/MOBI/Loc.h"
#include "MOF/MOBI/Module.h"
#include "MOF/MOBI/Symbol.h"
#include "MOF/MOBI/Type.h"

struct M_Attr {
  M::MOF::Builder::Attr attr;
};

struct M_AttrMap {
  M::MOF::Builder::AttrMap map;
};

struct M_Graph {
  M::MOF::Builder::Graph graph;
};

struct M_Loc {
  M::MOF::Builder::Loc loc;
};

struct M_Module {
  M::MOF::Builder::Module module;
};

struct M_Tuple {
  M::MOF::Builder::Tuple tuple;
};

struct M_Symbol {
  M::MOF::Builder::Symbol symbol;
};

struct M_Arity {
  M::MOF::Builder::Arity arity;
};

struct M_Type {
  M::MOF::Builder::Type type;
};

#endif // MOF_API_C_DETAIL_MOBI_H
