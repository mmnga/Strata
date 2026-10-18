#include "strata/deepseek41/flow_pool.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
void require(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
int main() {try {
    cpu_set_t original;CPU_ZERO(&original);require(!pthread_getaffinity_np(pthread_self(),sizeof(original),&original),"read caller affinity");
    std::vector<int> available;for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&original)) available.push_back(cpu);
    require(available.size()>=24,"pool clock test needs24 available logical CPUs");
    uint64_t epochs=0,jobs_done=0,hooks=0,hook_errors=0;
    for(int participants:{1,4,24}) for(int spin_us:{0,20000}) {
        std::vector<int> cpus(available.begin(),available.begin()+participants);
        {
            strata::deepseek41::FlowPool pool(cpus,spin_us);std::atomic<int> generation{-1};
            for(int epoch=0;epoch<1200;++epoch) {
                if(epoch%600==0) std::this_thread::sleep_for(std::chrono::milliseconds(25)); // Force idle/CV wake even with20ms default.
                const int jobs=(epoch*17)%75;std::vector<std::atomic<int>> claims(jobs),visits(participants);
                for(auto& v:claims) v.store(0);for(auto& v:visits) v.store(0);
                std::atomic<int> head{0};std::atomic<bool> bad{false};generation.store(epoch,std::memory_order_release);
                const bool with_hook=epoch%2==0,throws=with_hook && epoch%101==0;bool caught=false;
                const auto work=[&,epoch](int worker) {
                    if(generation.load(std::memory_order_acquire)!=epoch || worker<0 || worker>=participants) {bad.store(true);return;}
                    if(visits[worker].fetch_add(1)!=0) bad.store(true);
                    if(worker==participants-1 && epoch%97==0) std::this_thread::sleep_for(std::chrono::microseconds(50));
                    for(;;) {
                        const int job=head.fetch_add(1,std::memory_order_relaxed);if(job>=jobs) break;
                        if(generation.load(std::memory_order_acquire)!=epoch || claims[job].fetch_add(1)!=0) bad.store(true);
                        if(job%13==0) std::this_thread::yield();
                    }
                };
                const auto hook=[&] {++hooks;if(epoch%97==0) std::this_thread::sleep_for(std::chrono::microseconds(50));if(throws) throw std::runtime_error("intentional publish hook");};
                try {pool.run(work,with_hook ? std::function<void()>(hook) : std::function<void()>());}
                catch(const std::runtime_error&) {caught=true;++hook_errors;}
                require(caught==throws,"publish hook exception contract");require(!bad.load(),"stale or duplicate callback/job");
                for(const auto& v:visits) require(v.load()==1,"callback missing before run returned");
                for(const auto& v:claims) require(v.load()==1,"job missing before run returned");
                generation.store(-1,std::memory_order_release);++epochs;jobs_done+=jobs;
            }
        }
        require(!pthread_setaffinity_np(pthread_self(),sizeof(original),&original),"restore caller affinity");
    }
    require(epochs==7200 && hooks==3600,"pool fixture coverage");
    std::cout<<epochs<<" epochs /"<<jobs_done<<" jobs, 1/4/24 participants, default20ms+forcedCV, "<<hooks<<" publish hooks /"<<hook_errors
        <<" intentional errors, late workers, zero-job, exactly-once callback/job, idle wake/shutdown PASS\n";
}catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
