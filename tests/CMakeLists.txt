message(STATUS "Configuring tests of wavelet_rs")

add_executable(simple_cpp cpp/simple.cpp)
target_link_libraries(simple_cpp libwavelet)
set_property(TARGET simple_cpp PROPERTY CXX_STANDARD 11)