cmake_minimum_required(VERSION 3.0.2)
project(planning_module)

set(OMPL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/ompl)
message(STATUS "########################## OMPL_PATH: ${OMPL_PATH}")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        visualization_msgs
        nav_msgs
        tf
        grid_map_core
        grid_map_ros
        grid_map_msgs
        message_generation
        dynamic_reconfigure
)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

## Declare ROS messages
add_message_files(
  FILES
  PathData.msg  # 새로 만든 메시지 파일 추가
)

generate_messages(
  DEPENDENCIES
  visualization_msgs
  nav_msgs
  std_msgs  # Add this line
)

generate_dynamic_reconfigure_options(
  cfg/Hyperparameters.cfg 
)


catkin_package()



include_directories(
        /opt/ros/noetic/include
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${OMPL_PATH}/include
)

add_executable(main_planning src/main_planning.cpp)
target_link_libraries(main_planning
        ${catkin_LIBRARIES}
        ${Boost_LIBRARIES}
        ${Eigen_LIBRARIES}
        ${OMPL_PATH}/share/libompl.so
)


