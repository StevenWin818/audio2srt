SET(cargokit_cmake_root "${CMAKE_CURRENT_LIST_DIR}/..")

# Workaround for https://github.com/dart-lang/pub/issues/4010
get_filename_component(cargokit_cmake_root "${cargokit_cmake_root}" REALPATH)

if(WIN32)
    # REALPATH does not properly resolve symlinks on windows :-/
    execute_process(COMMAND powershell -ExecutionPolicy Bypass -File "${CMAKE_CURRENT_LIST_DIR}/resolve_symlinks.ps1" "${cargokit_cmake_root}" OUTPUT_VARIABLE cargokit_cmake_root OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Arguments
# - target: CMAKE target to which rust library is linked
# - manifest_dir: relative path from current folder to directory containing cargo manifest
# - lib_name: cargo package name
# - any_symbol_name: name of any exported symbol from the library.
#                    used on windows to force linking with library.
function(apply_cargokit target manifest_dir lib_name any_symbol_name)

    set(CARGOKIT_LIB_NAME "${lib_name}")
    set(CARGOKIT_LIB_FULL_NAME "${CMAKE_SHARED_MODULE_PREFIX}${CARGOKIT_LIB_NAME}${CMAKE_SHARED_MODULE_SUFFIX}")
    if (CMAKE_CONFIGURATION_TYPES)
        set(CARGOKIT_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        set(OUTPUT_LIB "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${CARGOKIT_LIB_FULL_NAME}")
    else()
        set(CARGOKIT_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
        set(OUTPUT_LIB "${CMAKE_CURRENT_BINARY_DIR}/${CARGOKIT_LIB_FULL_NAME}")
    endif()
    get_filename_component(CARGOKIT_MANIFEST_ABS "${CMAKE_CURRENT_SOURCE_DIR}/${manifest_dir}" REALPATH)
    set(CARGOKIT_TEMP_DIR "${CARGOKIT_MANIFEST_ABS}/target")

    if (FLUTTER_TARGET_PLATFORM)
        set(CARGOKIT_TARGET_PLATFORM "${FLUTTER_TARGET_PLATFORM}")
    else()
        if(CMAKE_HOST_WIN32)
            set(CARGOKIT_TARGET_PLATFORM "windows-x64")
        else()
            set(CARGOKIT_TARGET_PLATFORM "")
        endif()
    endif()

    set(CARGOKIT_ENV
        "CARGOKIT_CMAKE=${CMAKE_COMMAND}"
        "CARGOKIT_CONFIGURATION=$<CONFIG>"
        "CARGOKIT_MANIFEST_DIR=${CMAKE_CURRENT_SOURCE_DIR}/${manifest_dir}"
        "CARGOKIT_TARGET_TEMP_DIR=${CARGOKIT_TEMP_DIR}"
        "CARGOKIT_OUTPUT_DIR=${CARGOKIT_OUTPUT_DIR}"
        "CARGOKIT_TARGET_PLATFORM=${CARGOKIT_TARGET_PLATFORM}"
        "CARGOKIT_TOOL_TEMP_DIR=${CARGOKIT_TEMP_DIR}/tool"
        "CARGOKIT_ROOT_PROJECT_DIR=${CMAKE_SOURCE_DIR}"
        "CMAKE_GENERATOR_PLATFORM=" # 阻止 Ninja 下平台 x64 规格报错
        # CUDA 架构: 不在此设置 (llama.cpp ggml-cuda 默认即多架构列表:
        # native + 75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real,
        # 分发兼容不同 GPU；Debug/Release 行为一致)。
        "CMAKE_GENERATOR_TOOLSET="
    )

    if(CMAKE_HOST_WIN32)
        # 注意: 不要设置 CARGO_TARGET_DIR。cargokit build_tool 显式传
        # `--target-dir <manifest>/target` (命令行参数优先于 env)，
        # 系统 CMake 4.3.x 存在 CUDA native 探测回归 (llama.cpp GPU 探测失败:
        # "CUDA_ARCHITECTURES is set to native, but no NVIDIA GPU was detected")。
        # 仅当检测到回归版本时，才通过 CMAKE env 指定 VS 自带 cmake (无回归)；
        # 非 Windows 或 cmake 版本正常时不做任何干预，保持跨平台。
        set(_sys_cmake_ver "")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" --version
            OUTPUT_VARIABLE _sys_cmake_ver
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(_cmake_major "0")
        set(_cmake_minor "0")
        if(_sys_cmake_ver MATCHES "cmake version ([0-9]+)\\.([0-9]+)")
            set(_cmake_major "${CMAKE_MATCH_1}")
            set(_cmake_minor "${CMAKE_MATCH_2}")
        endif()
        set(_need_workaround OFF)
        if(_cmake_major GREATER 4 OR (_cmake_major EQUAL 4 AND _cmake_minor GREATER_EQUAL 3))
            set(_need_workaround ON)
        endif()
        if(_need_workaround)
            set(CMAKE_ENV_CMAKE "")
            foreach(_vs_root
                "C:/Program Files (x86)/Microsoft Visual Studio/19/BuildTools"
                "C:/Program Files/Microsoft Visual Studio/19/BuildTools"
                "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools"
                "C:/Program Files/Microsoft Visual Studio/18/BuildTools"
                "C:/Program Files/Microsoft Visual Studio/2022/Community"
                "C:/Program Files/Microsoft Visual Studio/2022/Professional"
                "C:/Program Files/Microsoft Visual Studio/2022/Enterprise"
                "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools")
                set(_cand "${_vs_root}/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
                if(EXISTS "${_cand}")
                    set(CMAKE_ENV_CMAKE "${_cand}")
                    break()
                endif()
            endforeach()
            if(NOT CMAKE_ENV_CMAKE STREQUAL "")
                list(APPEND CARGOKIT_ENV "CMAKE=${CMAKE_ENV_CMAKE}")
            endif()
        endif()
    endif()

    if (WIN32)
        set(SCRIPT_EXTENSION ".cmd")
        set(IMPORT_LIB_EXTENSION ".lib")
    else()
        set(SCRIPT_EXTENSION ".sh")
        set(IMPORT_LIB_EXTENSION "")
        execute_process(COMMAND chmod +x "${cargokit_cmake_root}/run_build_tool${SCRIPT_EXTENSION}")
    endif()

    # Using generators in custom command is only supported in CMake 3.20+
    if (CMAKE_CONFIGURATION_TYPES AND ${CMAKE_VERSION} VERSION_LESS "3.20.0")
        foreach(CONFIG IN LISTS CMAKE_CONFIGURATION_TYPES)
            add_custom_command(
                OUTPUT
                "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG}/${CARGOKIT_LIB_FULL_NAME}"
                "${CMAKE_CURRENT_BINARY_DIR}/_phony_"
                COMMAND ${CMAKE_COMMAND} -E env ${CARGOKIT_ENV}
                "${cargokit_cmake_root}/run_build_tool${SCRIPT_EXTENSION}" build-cmake
                VERBATIM
            )
        endforeach()
    else()
        add_custom_command(
            OUTPUT
            ${OUTPUT_LIB}
            "${CMAKE_CURRENT_BINARY_DIR}/_phony_"
            COMMAND ${CMAKE_COMMAND} -E env ${CARGOKIT_ENV}
            "${cargokit_cmake_root}/run_build_tool${SCRIPT_EXTENSION}" build-cmake
            VERBATIM
        )
    endif()


    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/_phony_" PROPERTIES SYMBOLIC TRUE)

    if (TARGET ${target})
        # If we have actual cmake target provided create target and make existing
        # target depend on it
        add_custom_target("${target}_cargokit" DEPENDS ${OUTPUT_LIB})
        add_dependencies("${target}" "${target}_cargokit")
        target_link_libraries("${target}" PRIVATE "${OUTPUT_LIB}${IMPORT_LIB_EXTENSION}")
        if(WIN32)
            target_link_options(${target} PRIVATE "/INCLUDE:${any_symbol_name}")
        endif()
    else()
        # Otherwise (FFI) just use ALL to force building always
        add_custom_target("${target}_cargokit" ALL DEPENDS ${OUTPUT_LIB})
    endif()

    # Allow adding the output library to plugin bundled libraries
    set("${target}_cargokit_lib" ${OUTPUT_LIB} PARENT_SCOPE)

endfunction()
