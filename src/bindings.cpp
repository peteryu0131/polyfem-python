// Minimal nanobind module entry point
// Note: binding.cpp already contains PY_MODULE(polyfempy, m)
// This file is kept as a backup/alternative entry point
// If using this file, comment out PY_MODULE in binding.cpp
#include "binding_wrapper.hpp"

// Uncomment below if you want to use this as the main entry point
// PY_MODULE(polyfempy, m) {
//     m.doc() = "PolyFEM Python bindings";
// }
