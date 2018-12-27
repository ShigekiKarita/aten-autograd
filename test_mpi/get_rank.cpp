#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>

namespace bmpi = boost::mpi;

int main(int argc, char** argv)
{
    bmpi::environment env(argc, argv);
    bmpi::communicator world;

    std::cout << "Name : " << env.processor_name() << std::endl;
    std::cout << "Rank : " << world.rank() << std::endl;
    std::cout << "Number of Node : " << world.size() << std::endl;
}
