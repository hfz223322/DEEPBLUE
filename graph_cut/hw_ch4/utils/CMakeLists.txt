
add_library(utils 
    display.cpp 
    tictoc.cpp
    opencv_utils.cpp
)
target_link_libraries(utils ${OpenCV_LIBS})

add_library(distribution distribution.cpp)
target_link_libraries(distribution gmm ${OpenCV_LIBS})

add_library(em em.cpp)

add_library(gmm gmm.cpp)
target_link_libraries(gmm em)