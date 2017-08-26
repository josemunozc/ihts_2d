CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.5.0 REQUIRED
  HINTS /opt/dealii/
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(ihts)

INCLUDE_DIRECTORIES(/home/c1045890/code/libraries)

add_definitions("-Wall")

ADD_LIBRARY(mylib /home/c1045890/code/libraries/SurfaceCoefficients.cpp /home/c1045890/code/libraries/AnalyticSolution.cpp /home/c1045890/code/libraries/BoundaryConditions.cpp /home/c1045890/code/libraries/DataTools.cpp /home/c1045890/code/libraries/Names.cpp /home/c1045890/code/libraries/Material.cpp /home/c1045890/code/libraries/PorousMaterial.cpp)
DEAL_II_SETUP_TARGET(mylib)


ADD_EXECUTABLE(mycode ihts_2d.cc)
DEAL_II_SETUP_TARGET(mycode)

TARGET_LINK_LIBRARIES(mycode mylib)


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
