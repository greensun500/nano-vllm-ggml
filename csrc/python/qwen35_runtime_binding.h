#pragma once

#include <Python.h>

namespace nanovllm::python {

// Registers nanovllm._C.Qwen35Runtime.  Returns 0 on success and -1 with a
// Python exception set on failure.  The module keeps the type's owning
// reference after a successful call.
int register_qwen35_runtime_type(PyObject * module);

}  // namespace nanovllm::python
