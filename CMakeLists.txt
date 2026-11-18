cmake_minimum_required(VERSION 3.10)

project(zhang_calib
        VERSION 1.0
        DESCRIPTION "Zhang's camera calibration, pure C++ with no dependencies"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
#  C++17，打开优化
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W3 /utf-8)
else()
    add_compile_options(-Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
#  可执行文件
# ---------------------------------------------------------------------------
add_executable(zhang_calib
    src/main.cpp
    src/zhang.cpp)

target_include_directories(zhang_calib PRIVATE src)

# 生成的程序放在项目根目录下的 bin/
set_target_properties(zhang_calib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# ---------------------------------------------------------------------------
#  安装
# ---------------------------------------------------------------------------
install(TARGETS zhang_calib RUNTIME DESTINATION bin)
