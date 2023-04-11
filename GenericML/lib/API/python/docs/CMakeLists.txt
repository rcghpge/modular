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

  add_custom_target(
    inference_engine_python_docs_postprocessing_cleanup
    DEPENDS docs-inference_engine_python-html
    COMMAND ${CMAKE_COMMAND} -E rm -rf _static _sources
    COMMAND ${CMAKE_COMMAND} -E rm -f objects.inv searchindex.js
    COMMAND ${CMAKE_COMMAND} -E rm -f genindex.html search.html py-modindex.html index.html
    COMMAND ${CMAKE_COMMAND} -E rm -f genindex.md search.md py-modindex.md index.md
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/postprocess.py -d ${CMAKE_CURRENT_BINARY_DIR}

    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Remove unwanted directories and files from the sphinx-build."
    VERBATIM
  )

  if(TARGET inference_engine_docs)
    add_dependencies(inference_engine_docs inference_engine_python_docs_postprocessing_cleanup)
  endif()
else()
  message("Skipping doc generation for Inference Engine Python APIs
           since `sphinx` is not installed.")
endif()
