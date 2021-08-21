#include "gradient_descent_base.h"
#include <iostream>

class my_gradient_descent : public GradientDescentBase
{
    public:
    my_gradient_descent(double step_size, double init_value): GradientDescentBase(step_size), x_init(init_value){}

    protected:
    void initialize() override;
    void update() override;
    double compute_energy() override;
    void roll_back_state() override;
    void back_up_state() override;
    std::string return_drive_class_name() const override;

    private:
    double x, x_init, last_x;
};

void my_gradient_descent::initialize()
{
    x = x_init;
}

void my_gradient_descent::update()
{
    x = x/(1.0 + step_size_);
    std::string a;
    getline(std::cin, a);
}

double my_gradient_descent::compute_energy()
{
    double y = 0.5 * x * x;
    return y;
}

void my_gradient_descent::roll_back_state()
{
    x = last_x;
}

void my_gradient_descent::back_up_state()
{
    last_x = x;
}

std::string my_gradient_descent::return_drive_class_name() const
{
    return "my_gradient_descent";
}

int main()
{
    double step_size = 0.01;
    double init_val = 100;
    int iters = 10000;
    my_gradient_descent mygd(step_size, init_val);
    mygd.run(iters);
    return 0;
}