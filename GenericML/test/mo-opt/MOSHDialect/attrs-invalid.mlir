// RUN: mo-opt %s -verify-diagnostics

// expected-error @below{{expected all elements to have the same type}}
#foo = #mosh<ape[:i64 1, :i8 ?, :index N]> : !mosh.ape
