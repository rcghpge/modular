//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "StandardStatus.h"

const char *standardStatusMessages[M_DRIVER_NUM_STANDARD_STATUS_CODES] = {
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
};
