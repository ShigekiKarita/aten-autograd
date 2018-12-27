#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/collectives.hpp>
#include <iostream>
#include <ctime>
#include <cstdlib>

namespace bmpi = boost::mpi;

int main(int argc, char** argv)
{
    bmpi::environment env(argc, argv);
    bmpi::communicator world;
    std::cout << "Node : " << env.processor_name()
              << ", Rank : " << world.rank()
              << "/" << world.size() << std::endl;

    std::vector<double> x;
    std::vector<double> y;

    if(world.rank() == 0)
    {
        std::srand(std::time(0));
        x.resize(world.size());
        y.resize(world.size());
        auto f = []() { return double(std::rand()) / RAND_MAX; };
        std::generate(std::begin(x), std::end(x), f);
        std::generate(std::begin(y), std::end(y), f);
    }

    double x_value;
    double y_value;
    bmpi::scatter(world, x, x_value, 0);
    bmpi::scatter(world, y, y_value, 0);

    const double result = x_value + y_value;
    bmpi::gather(world, result, x, 0);

    if(world.rank() == 0)
    {
        std::cout << "Result." << std::endl;;
        std::for_each(std::begin(x), std::end(x), [](double i) { std::cout << i << std::endl; });
    }
}
