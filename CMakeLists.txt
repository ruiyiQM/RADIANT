cmake_minimum_required(VERSION 3.16)
project(RADIANT LANGUAGES NONE)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy, scipy, pyscf; assert tuple(map(int, pyscf.__version__.split('.')[:2])) >= (2, 12)"
        RESULT_VARIABLE dependency_status ERROR_VARIABLE dependency_error)
    if(NOT dependency_status EQUAL 0)
        message(FATAL_ERROR "Regression environment requires NumPy, SciPy and PySCF >=2.12: ${dependency_error}")
    endif()
    foreach(suite numerics propagation pt_cn pt_mixed pt_im adiabatic pyscf execution mash input)
        add_test(NAME ${suite} COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_${suite}.py" -v)
        set_tests_properties(${suite} PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src;PYTHONDONTWRITEBYTECODE=1;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            LABELS "regression" TIMEOUT 120)
    endforeach()
endif()
