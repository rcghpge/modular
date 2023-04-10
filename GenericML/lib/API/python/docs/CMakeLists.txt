##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

include(FindSphinx)

if(Sphinx_FOUND)
  configure_file(conf.py.in conf.py @ONLY)

  include(AddModularSphinxTarget)
  add_modular_sphinx_target(html inference_engine_python
    SOURCE_DIR
    ${CMAKE_SOURCE_DIR}/GenericML/lib/API/python/modular/engine
    CONF_DIR
    ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Have doc-generation internal CMake target depend on the Python wheel,
  # This ensures that doc generation is always using the up-to-date wheel and
  # avoids any surprises in `autodoc` failing to import the Python wheel in the
  # case it is not built yet.
  add_dependencies(docs-inference_engine_python-html modular-pybind-ext)
else()
  message("Skipping doc generation for Inference Engine Python APIs
           since `sphinx` is not installed.")
endif()
