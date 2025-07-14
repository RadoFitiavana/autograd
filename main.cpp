#include <iostream>
#include <vector>
#include <iomanip>
#include "autoFun.h"

using namespace autoDiff;

template <typename T>
void print_vector(const std::string& label, const std::vector<T>& vec)
{
    std::cout << label << ": [";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << std::fixed << std::setprecision(6) << vec[i];
        if (i < vec.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

template <typename T>
std::vector<T> make_input(size_t n, T base = 1.0)
{
    std::vector<T> input(n);
    for (size_t i = 0; i < n; ++i)
        input[i] = base + static_cast<T>(i);  // e.g., [1, 2, 3, ..., n]
    return input;
}

int main()
{
    using T = double;
    std::cout << "==== AutoDiff Test Cases ====\n";

    // Test 1: Sum of squares f(x) = x1^2 + x2^2 + ... + xn^2
    {
        size_t n = 5;
        std::cout << "\n[Test 1] f(x) = sum(i * x_i^2), x ∈ R^" << n << "\n";

        AutoFunction<T> f(n, [](const std::vector<var<T>>& vars) -> var<T> {
            var<T> result = cg::make_var(0.0);
            T index = 1.0;
            for (const var<T>& node : vars)
            {
                result = result + node * index * node;
                index += 1.0;
            }
            return result;
            });

        std::vector<T> input = make_input<T>(n);  // [1, 2, 3, 4, 5]
        print_vector("Input", input);

        T value = f(input);
        auto grad_forward = f.grad(input, false);
        auto grad_reverse = f.grad(input, true);

        std::cout << "Value: " << value << "\n";
        print_vector("Forward Gradient", grad_forward);
        print_vector("Reverse Gradient", grad_reverse);
    }

    // Test 2: f(x) = x1 * x2 + x2 * x3 + ... + x_{n-1} * x_n
    {
        size_t n = 6;
        std::cout << "\n[Test 2] f(x) = sum(x_i * x_{i+1}), x ∈ R^" << n << std::endl ;

        AutoFunction<T> f(n, [](const std::vector<var<T>>& vars) -> var<T> {
            var<T> result = cg::make_var(0.0);
            for (size_t i = 0; i < vars.size() - 1; ++i)
            {
                result = result + vars[i] * vars[i + 1];
            }
            return result;
            });

        std::vector<T> input = make_input<T>(n);  // [1, 2, 3, 4, 5, 6]

        print_vector("Input", input);

        T value = f(input);
        auto grad_forward = f.grad(input, false);
        auto grad_reverse = f.grad(input, true);

        std::cout << "Value: " << value << "\n";
        print_vector("Forward Gradient", grad_forward);
        print_vector("Reverse Gradient", grad_reverse);
    }

    // Test 3: Nonlinear function f(x) = sin(x1) + log_10(x2) + exp(x3) + pow(x2,5)
    {
        size_t n = 3;
        std::cout << "\n[Test 3] f(x) = sin(x1) + log_10(x2) + exp(x3) + x2^5, x ∈ R^3\n";

        AutoFunction<T> f(n, [](const std::vector<var<T>>& vars) -> var<T> {
            var<T> sinx = node::sin(vars[0]);
            var<T> logx = node::log(vars[1], 10.0);
            var<T> expx = node::exp(vars[2]);
            var<T> powx5 = node::pow(vars[1], 5);

            return sinx + logx + expx + powx5 ;
            });

        std::vector<T> input = { 0.5, 2.0, 1.0 };

        print_vector("Input", input);

        T value = f(input);
        auto grad_forward = f.grad(input, false);
        auto grad_reverse = f.grad(input, true);

        std::cout << "Value: " << value << "\n";
        print_vector("Forward Gradient", grad_forward);
        print_vector("Reverse Gradient", grad_reverse);
    }

    std::cout << "\n==== End of Tests ====\n";
    return 0;
}
