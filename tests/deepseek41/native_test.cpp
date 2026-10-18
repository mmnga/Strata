#include "fixture.hpp"
#include "strata/deepseek41/engine.hpp"
#include <iostream>
#include <unistd.h>

int main() {
    const auto path = std::filesystem::temp_directory_path()/"strata-deepseek41-native-test.gguf";
    try {
        deepseek_fixture::write(path,true);
        int fd[2]; if (pipe(fd)) throw std::runtime_error("pipe");
        std::string commands = "CHECK ";
        for (int i = 0; i < 136; ++i) { if (i) commands += ','; commands += std::to_string(i%16); }
        commands += "\nGEN 24 temperature=0 0,1,2,3,4,5,6,7\nQUIT\n";
        if (write(fd[1],commands.data(),commands.size()) != ssize_t(commands.size())) throw std::runtime_error("write");
        close(fd[1]); dup2(fd[0],STDIN_FILENO); close(fd[0]);
        std::vector<std::string> args = {"strata-deepseek41","--native",path.string(),"--tp","2","--ctx","256","--threads","24","--expert-cache-mib","1","--lookahead","2","--serve"};
        std::vector<char*> argv; for (auto& a : args) argv.push_back(a.data());
        const int result = strata::deepseek41::run(int(argv.size()),argv.data());
        std::filesystem::remove(path); return result;
    } catch (const std::exception& e) { std::filesystem::remove(path); std::cerr << e.what() << '\n'; return 1; }
}
