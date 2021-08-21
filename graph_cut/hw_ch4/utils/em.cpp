
#include "em.h"
#include <iostream>

// EM::EM() {
// }

void EMBase::run(int max_iteration) {
    initialize();
    for (int i = 0; i < max_iteration; i++) {
        std::cout << "current iteration : " << i << '\n';
        update_e_step();
        update_m_step();
    }
    print_terminate_info();
}

void EMBase::print_terminate_info() const {
    std::cout << "Iteration finished" << std::endl;
}
