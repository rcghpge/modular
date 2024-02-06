##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

if (NOT TARGET inference_engine_docs)
  return()
endif()

include(FindSphinx)

if(Sphinx_FOUND)
  configure_file(conf.py.in conf.py @ONLY)

  set(sphinx_builder html)
  # TODO(scott): Remove use of MODULAR_DOCSITE_V2 after docsite migration
  if (MODULAR_BUILD_INFERENCE_ENGINE_DOCS_PREVIEW)
    set(sphinx_output_dir ${CMAKE_CURRENT_BINARY_DIR}/preview)
  else()
    set(sphinx_output_dir ${CMAKE_CURRENT_BINARY_DIR}/web)
    if (MODULAR_DOCSITE_V2)
      set(sphinx_builder markdown)
    endif()
  endif()

  add_custom_target(inference_engine_python_docs_sphinx
    COMMAND
      ${SPHINX_EXECUTABLE}
      -b ${sphinx_builder}
      -c ${CMAKE_CURRENT_BINARY_DIR}
      -q # quiet
      -W # treat warnings as errors for now
      ${CMAKE_SOURCE_DIR}/SDK/lib/EngineAPI/python/max/engine # Source
      ${sphinx_output_dir} # Output
    DEPENDS max-pybind-ext # ensure the wheel is always up-to-date
    COMMENT "Generating Sphinx documentation for AI Engine Python API into ${sphinx_output_dir}."
    VERBATIM
  )

  add_custom_target(inference_engine_docs_python)

  # We only want to do cleanup of files in preview mode
  if (MODULAR_BUILD_INFERENCE_ENGINE_DOCS_PREVIEW)
    add_dependencies(inference_engine_docs_python inference_engine_python_docs_sphinx)
  else()
    add_custom_target(
      inference_engine_docs_python_postprocessing_cleanup
      DEPENDS inference_engine_python_docs_sphinx
      COMMAND ${CMAKE_COMMAND} -E rm -rf _static _sources
      COMMAND ${CMAKE_COMMAND} -E rm -f objects.inv searchindex.js
      COMMAND ${CMAKE_COMMAND} -E rm -f genindex.html search.html py-modindex.html index.html
      COMMAND ${CMAKE_COMMAND} -E rm -f genindex.md search.md py-modindex.md index.md
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/postprocess.py -d ${CMAKE_CURRENT_BINARY_DIR} -s ${MODULAR_DOCSITE_V2}

      WORKING_DIRECTORY ${sphinx_output_dir}
      COMMENT "Remove unwanted directories and files from the sphinx-build."
      VERBATIM
    )
    add_dependencies(inference_engine_docs_python inference_engine_docs_python_postprocessing_cleanup)
  endif()

  add_dependencies(inference_engine_docs inference_engine_docs_python)
else()
  message("Skipping doc generation for AI Engine Python APIs
           since `sphinx` is not installed.")
endif()
