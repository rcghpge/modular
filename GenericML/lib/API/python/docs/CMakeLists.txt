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
else()
  message("Skipping doc generation for Inference Engine Python APIs
           since `sphinx` is not installed.")
endif()
