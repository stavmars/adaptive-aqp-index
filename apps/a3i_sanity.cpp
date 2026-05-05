// a3i_sanity: prints the project version. Minimal smoke executable
// confirming the build wires headers and an app target together.
//
// Usage:
//   a3i_sanity            print version and exit 0
//   a3i_sanity --help     show this message

#include <cstring>
#include <iostream>

#include "a3i/util/version.hpp"

namespace {

void print_usage(std::ostream& os) {
    os << "Usage: a3i_sanity [--help]\n"
       << "  Prints the a3i version string and exits.\n";
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(std::cout);
            return 0;
        }
        std::cerr << "a3i_sanity: unknown argument '" << argv[i] << "'\n";
        print_usage(std::cerr);
        return 2;
    }
    std::cout << "a3i " << a3i::version() << "\n";
    return 0;
}
