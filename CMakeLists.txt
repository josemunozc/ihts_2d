CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.5.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(mycode)

INCLUDE_DIRECTORIES($ENV{HOME}/git/libraries)
ADD_LIBRARY(mylib 
    $ENV{HOME}/git/libraries/SurfaceCoefficients.cpp 
    $ENV{HOME}/git/libraries/AnalyticSolution.cpp 
    $ENV{HOME}/git/libraries/BoundaryConditions.cpp 
    $ENV{HOME}/git/libraries/DataTools.cpp 
    $ENV{HOME}/git/libraries/Names.cpp 
    $ENV{HOME}/git/libraries/Material.cpp 
    $ENV{HOME}/git/libraries/PorousMaterial.cpp)

add_definitions("-Wall")

DEAL_II_SETUP_TARGET(mylib)


ADD_EXECUTABLE(${PROJECT_NAME} ihts_2d.cc)
DEAL_II_SETUP_TARGET(${PROJECT_NAME})

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC $ENV{HOME}/git/libraries)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} mylib)


ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

ADD_CUSTOM_TARGET(run
  COMMAND mpirun -np 4 mycode input.prm
  )
