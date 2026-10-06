//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "StandardStatus.h"

const char *standardStatusMessages[M_DRIVER_NUM_STANDARD_STATUS_CODES] = {
    /*0*/ "Success",
    /*1*/ "Incompatible version",
    /*2*/ "Unknown error",
    /*3*/ "Argument out of range",
    /*4*/ "Unknown property name",
    /*5*/ "No devices",
    /*6*/ "Bad driver",
    /*7*/ "Bad device",
    /*8*/ "Bad context",
    /*9*/ "Bad memory",
    /*10*/ "Bad queue",
    /*11*/ "Bad event",
    /*12*/ "Bad function",
    /*13*/ "Out of memory",
    /*14*/ "Invalid argument",
    /*15*/ "Incompatible file type",
    /*16*/ "Bad bundle",
};
