add_executable(test_graph_cut test_graph_cut.cpp)
target_link_libraries(test_graph_cut
   utils
   ${OpenCV_LIBS}
   image_graph
   interaction_tool
   graph_cut
)


