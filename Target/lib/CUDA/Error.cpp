//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Error.h"

const char *statusMessages[M_DRIVER_NUM_ERRORS] = {
    /*0*/ "Success",
    /*1*/ "Unknown error",
    /*2*/ "Argument out of range",
    /*3*/ "Unknown property name",
    /*4*/ "No devices",
    /*5*/ "Bad device",
    /*6*/ "Bad context",
    /*7*/ "Bad memory",
    /*8*/ "Out of memory",
    /*9*/ "Invalid argument",
    /*10*/ "Unimplemented",
    /*11*/ "Bad handle",
};
