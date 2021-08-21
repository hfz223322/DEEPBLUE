
add_library(graph graph.cpp)
# target_link_libraries(graph graph_search)
add_library(image_graph image_graph.cpp)
target_link_libraries(image_graph graph distribution ${OpenCV_LIBS})



add_library(interaction_tool interaction_tool.cpp)
target_link_libraries(interaction_tool ${OpenCV_LIBS} utils)

add_library(graph_cut graph_cut.cpp)
target_link_libraries(graph_cut interaction_tool image_graph)