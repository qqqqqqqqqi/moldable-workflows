#include <cstdlib>
#include <cstdio>
#include <limits>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <utility>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <functional>
#include <omp.h>
#include <queue>
#include <filesystem>
#include <getopt.h>
#include <random>
#include <unordered_set>
#include "Trace.hpp"

#define INF std::numeric_limits<double>::max();


// #define  DEBUG
#pragma omp declare reduction (+: std::vector<int>: \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(),std::plus<int>())) \
initializer(omp_priv = std::vector<int>(omp_orig.size(), 0))
#pragma omp declare reduction (+: std::vector<double>: \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()))\
initializer(omp_priv = std::vector<double>(omp_orig.size(), 0))
const int PROC_CORES = omp_get_num_procs();
std::vector<double> seeds;

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
        tokens.push_back(item);
    return tokens;
}

std::map<std::string, std::string> code2strategy = {
    {"0", "ckptAll"}, {"1", "ckptCrossover"}, {"4", "our strategy"}, {"6", "ckptNone"},{"3","ckptNoReduce"}
};

struct param_t {
    int sample;
    std::string indir;
    std::string outdir;
    std::string dpddir;
    std::vector<std::string> Cs; //ckpt cost
    std::vector<std::string> Vs; //verification cost
    std::vector<std::string> pfails;
    std::vector<std::string> workflows;
    std::vector<std::string> P;
    std::vector<std::string> strates;
    std::vector<std::string> strates_name;
    std::vector<std::string> chain;
    int mode;
};

struct node_t {
    long id;
    double weight;
    std::string label;
    std::vector<std::pair<long, std::pair<long, double> > > inputs;
    std::vector<std::pair<long, std::pair<long, double> > > outputs;
    bool checkpoint;
    int proc;
    bool done;
    bool failed;

    double R;
    double C;
    double V;

    int nproc; //number of processors allocated to each task
    std::vector<int> procs; //processors allocated to each task
};

struct graph_t {
    long nb_nodes; //total number of nodes in the graph
    std::vector<node_t> nodes;
    std::vector<std::vector<long> > schedule; //The list scheduling of the graph
    std::vector<long> inDegree; //indegree of each node
    std::vector<std::vector<long>> succs; //successors[id] of each node
    bool ckpt;
    double makespan;
    double lam;
};

struct simulator_t {
    double start;
    double horizon; //the stop time
    std::vector<Trace> silent_errors;
    long global_nF; //number of silent errors for the platform
    double muF; //mu of silent error
};

struct result_t {
    double time;
    double nb_faults;
};

void initSimulator(int nprocs, double lambda, double horizon, simulator_t *s) {
    s->start = 0;
    s->horizon = horizon;

    // Convert lambda to MTBF in year
    if (lambda == 0) s->muF = s->horizon * 1000000;
    else s->muF = (1.0 / lambda) / ONEYEAR;

    //Initialize the lists of errors for each processor
    s->global_nF = 0;
    for (int i = 0; i < nprocs; i++) {
        s->silent_errors.emplace_back(s->muF,  s->horizon);
        s->global_nF += s->silent_errors[i].nF;
    }
}

void readDependencies(graph_t *G,std::map<std::string, long> &dict,const std::string &dfile,double ccr) {
    long file_id=0;
    std::unordered_map<std::string, long> file2id;
    std::ifstream input(dfile, std::ios::in);
    if (!input.is_open()) {
        throw std::runtime_error("Error opening file:" +dfile);
    }
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::vector<std::string> split_elems;
        split_elems = split(line, ',');
        for (unsigned i = 2; i < split_elems.size(); i += 2) {
            long fd;
            if(file2id.contains(split_elems[i])) {
                fd=file2id[split_elems[i]];
            }else {
                file2id[split_elems[i]]=file_id;
                fd=file_id;
                file_id++;
            }
            G->nodes[dict[split_elems[0]]].outputs.emplace_back(dict[split_elems[1]],
                                                                std::pair<long, double>(
                                                                    fd, std::stod(split_elems[i + 1])*ccr));
            G->nodes[dict[split_elems[1]]].inputs.emplace_back(dict[split_elems[0]],
                                                               std::pair<long, double>(
                                                                   fd ,std::stod(split_elems[i + 1])*ccr));
        }
    }
}

void readGraph(graph_t *G, const std::string& filename, int *n,const std::string &dfile,double ccr) {
    std::map<std::string, long> dict;
    std::ifstream input(filename, std::ios::in);
    if (!input.is_open()) {
        throw std::runtime_error("Error opening file:" +filename);
    }
    std::string line;
    std::getline(input, line);
    std::vector<std::string> split_elems;
    long id = 0;
    while (std::getline(input, line)) {
        if (line.substr(0, 2) == "0,")
            break;
        split_elems = split(line, ',');
        dict.insert(std::pair<std::string, long>(split_elems[0], id));
        id++;
        node_t node;
        node.label = split_elems[0];
        node.id = id - 1;
        node.weight = std::stod(split_elems[1], nullptr);
        std::vector<std::string> sProcs = split(split_elems[2], '_');
        std::ranges::transform(sProcs, std::back_inserter(node.procs),
                               [](const std::string &str) { return std::stoi(str); });
        node.nproc = node.procs.size();
        node.checkpoint = std::stoi(split_elems[3], nullptr) == 1;
        node.C = std::stod(split_elems[4]);
        node.V = std::stod(split_elems[5]);
        node.done = false;
        node.failed = false;
        G->nodes.push_back(node);
    }
    G->nb_nodes = id;

    std::vector<std::vector<long> > sched;
    int p = 0;
    do {
        sched.emplace_back();
        std::vector<std::string> split_elems = split(line, ',');
        for (auto i = 1; i < (int) split_elems.size(); i++)
            sched[p].push_back(dict[split_elems[i]]);
        p++;
    } while (std::getline(input, line) && line != "makespan,lambda");
    G->schedule = sched;
    getline(input, line);
    std::vector<std::string> el = split(line, ',');
    G->makespan = std::stod(el[0]);
    G->lam = std::stod(el[1]);
    input.close();
    readDependencies(G,dict,dfile,ccr);
    *n = p;
}

std::vector<double> computeBusySlots(int nprocs,graph_t *G) {
    std::vector<double> Bp(nprocs,0.0);
    for (int i = 0; i < nprocs; ++i) {
        for (long tid:G->schedule[i]) {
            Bp[i]+=G->nodes[tid].weight/ONEYEAR;
        }
    }
    return Bp;
}

std::vector<double> preComputePb(int nprocs,graph_t *G,std::vector<double> &Bp,double lambda) {
    std::vector<double> qp(nprocs, 0.0);
    for (int i = 0; i < nprocs; ++i) {
        double x= lambda*Bp[i];
        qp[i] = 1.0 - exp(-x);
        if (qp[i] < 0) qp[i] = 0;
        if (qp[i] > 1) qp[i] = 1;
    }
    return qp;
}

void startMCSimulationNoCkpt(int nprocs,simulator_t *s, result_t *r, graph_t *G) {
    double t=0.0;
    double sinkV=0.0;
    double sinkC=0.0;
    for(auto &node: G->nodes) {
        if(node.V!=0) {
            sinkV += node.V/ONEYEAR;
            sinkC += node.C/ONEYEAR;
        }
    }
    double base_fail= (G->makespan+sinkV)/ONEYEAR;
    double base_success=(G->makespan+sinkV+sinkC)/ONEYEAR;
    std::vector<double> Bp=computeBusySlots(nprocs,G);
    std::vector<double> qp=preComputePb(nprocs,G,Bp,1/s->muF);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int nb_faults=0;
    while(true) {
        bool anyError=false;
        for (int i = 0; i < nprocs; ++i) {
            double u = (double)rand() / (double)RAND_MAX;
            if (qp[i]>0.0&&u<qp[i]) {
                anyError=true;
                nb_faults++;
            }
        }
        if(anyError) {
            t+=base_fail;
            if(t>s->horizon) {
                break;
            }
        }else {
            t+=base_success;
            break;
        }
    }
    r->nb_faults=nb_faults;
    if (t>s->horizon) {
        r->time=s->horizon*ONEYEAR/ONEHOUR;
    }else {
       r->time=t*ONEYEAR/ONEHOUR;
    }
}

void initG(graph_t *G) {
    long nb=G->nb_nodes;
    G->succs=std::vector<std::vector<long>>(nb);
    G->inDegree=std::vector<long>(nb,0);
    for(int i=0;i<nb;i++) {
        G->inDegree[i]=(int)G->nodes[i].inputs.size();
        for(const auto& succ:G->nodes[i].outputs) {
            G->succs[i].push_back(succ.first);
        }
    }
}

void startSimulationSF(int nprocs, simulator_t *s, result_t *r, graph_t *G) {
    std::vector<double> cTime = std::vector(nprocs, s->start);
    std::vector<long> scheduled = std::vector<long>(nprocs, -1);
    const auto& listScheduling = G->schedule;
    std::vector<int> currentTask = std::vector<int>(nprocs, -1);
    //lastCkpt[p] will contain the index in listScheduling[p] of the last checkpointed task for processor p
    std::vector<int> lastCkpt = std::vector<int>(nprocs, -1);
    std::vector<bool> errorState = std::vector<bool>(nprocs, false);
    std::vector<long> nextTasks(nprocs, -1);
    double global_time = s->start;
    std::vector<long> inDegree=G->inDegree;
    int exec = 0;
    int sF = 0;
    std::unordered_map<long, std::unordered_set<long> > memData; //file:<outputs>
    std::vector<std::unordered_set<long>> pMem(nprocs);
    do {
#ifdef DEBUG
        std::cout << "===Time : " << global_time << "===\n";
#endif
        /****************************
        BEGIN SCHEDULE
        ****************************/
        std::vector<unsigned char> unvisited = std::vector<unsigned char>(nprocs, 1);
        for (int i = 0; i < nprocs; i++) {
            if (unvisited[i] && scheduled[i] != -1 && global_time >= cTime[i]) {
                node_t *node = &G->nodes[scheduled[i]];
                std::vector<int> processors = node->procs;
                if (node->failed) {
                    for (const auto &p: processors) {
                        for (int k = lastCkpt[p] + 1; k < currentTask[p]; k++) {
                            if (G->nodes[listScheduling[p][k]].done) {
                                exec--;
                                G->nodes[listScheduling[p][k]].done = false;
                                G->nodes[listScheduling[p][k]].failed = false;
                                for(auto succ:G->succs[listScheduling[p][k]]) {
                                    inDegree[succ]++;
                                }

#ifdef DEBUG
                                    std::cout << G->nodes[listScheduling[p][k]].label << " rollback need to re-execute\n";
#endif
                            }
                        }
                        errorState[p] = false;
                        currentTask[p] = lastCkpt[p]; //NEED TO ROLLBACK
                        unvisited[p] = 0;
                    }
                    node->failed = false;
                } else {
                    node->done = true;
                    exec++;
                    for(auto succ:G->succs[node->id]) {
                        inDegree[succ]--;
                    }
#ifdef DEBUG
                        std::cout << "task " << node->label << " done.\n";
                        std::cout << "Executed : " << exec << "\n";
#endif
                    for (const auto &p: processors) {
                        unvisited[p] = 0;
                    }
                }
            }
        }

        /****************************
             UPDATE READY TASKS
        ****************************/
        std::fill(nextTasks.begin(), nextTasks.end(), -1);
        for (int i = 0; i < nprocs; i++) {
            if (currentTask[i] < (int) listScheduling[i].size() - 1)
                nextTasks[i] = listScheduling[i][currentTask[i] + 1];
        }
        for (int i = 0; i < nprocs; i++) {
            if (cTime[i] <= global_time) {
                //If cTime[i]>global time : task on processor i is still running
                if (nextTasks[i] != -1) {
                    long next_task = nextTasks[i];
                    if(inDegree[next_task] == 0) {
                        bool free = true;
                        for (auto p: G->nodes[next_task].procs) {
                            if (cTime[p] > global_time || next_task != nextTasks[p]) {
                                // Require that other processors are idle and the pending tasks are the same
                                free = false;
                                break;
                            }
                        }
                        if (free) {
                            currentTask[i]++;
                            scheduled[i] = next_task;
                            cTime[i] = global_time;
                        } else {
                            //processor begin free
                            scheduled[i] = -1;
                        }
                    } else {
                        //processor begin free
                        scheduled[i] = -1;
                    }
                } else {
                    // no next task
                    scheduled[i] = -1;
                }
            }
        }

        /****************************
        IF SILENT ERROR STRIKES
        ****************************/
        std::unordered_map<long,double> rv;
        for (int i = 0; i < nprocs; i++) {
            if (scheduled[i] != -1 && cTime[i] <= global_time) {
                /** READ **/
                long node_id=scheduled[i];
                node_t *node = &G->nodes[scheduled[i]];
                double R = 0;
                if (!rv.contains(node_id)) {
                    //read from memory or read from stable storage
                    for (const auto &ret: node->inputs) {
                        if (!memData.contains(ret.second.first)) {
                                if(!pMem[i].contains(ret.second.first)) {//need to read from stable storage
                                    R+=ret.second.second;
                                    pMem[i].insert(ret.second.first);
                                }
                        } else {
                            memData[ret.second.first].erase(node_id);
                            if (memData[ret.second.first].empty()) {
                                //if all successors read the file ,delete!
                                memData.erase(ret.second.first);
                            }
                        }
                    }
                    rv[node_id] = R;
                } else {
                    R = rv[node_id];
                }
                /** If silent error strikes **/

                double weight = G->nodes[scheduled[i]].weight;
                weight /= ONEYEAR;
                R /= ONEYEAR;
                double nextError = s->silent_errors[i].next(cTime[i] + R);
                //silent error only strikes computation
                if (nextError < cTime[i] + R + weight) {
                    errorState[i] = true;
                    sF++;
#ifdef DEBUG
                    std::cout << G->nodes[scheduled[i]].label << " suffer silent error in processor " << i << " at " <<
                            nextError << "\n";
#endif
                }
            }
        }

        /**
        *VERIFICATION / MEMORY CHECKPOINT
        **/
        std::fill(unvisited.begin(), unvisited.end(), 1);
        for (int i = 0; i < nprocs; i++) {
            if (unvisited[i] && scheduled[i] != -1 && cTime[i] <= global_time) {
                node_t *node = &G->nodes[scheduled[i]];
                const std::vector<int> processors = node->procs;
                double R = rv[node->id];

                if (node->checkpoint) {
                    bool rollBack = false;
                    for (const auto &p: processors) {
                        //Verification if need to rollback
                        cTime[p] += (R / ONEYEAR + node->weight / ONEYEAR + node->V / ONEYEAR);
                        unvisited[p] = 0;
                        pMem[p].clear();
                        if (errorState[p]) {
                            rollBack = true;
                        }
                    }
                    if (rollBack) {
                        node->failed = true;
                    } else {
                        double C = node->C / ONEYEAR;
                        for (const auto &p: processors) {
                            errorState[p] = false;
                            cTime[p] += C;
                            lastCkpt[p] = currentTask[p];
                        }
                        for (const auto &ret: node->outputs) {
                            if (memData.contains(ret.second.first)) {
                                memData[ret.second.first].insert(ret.first);
                            } else {
                                memData.insert(std::make_pair(ret.second.first, std::unordered_set<long>{ret.first}));
                            }
                        }
                    }
                } else {
                    for (const auto &p: processors) {
                        cTime[p] += (R / ONEYEAR + node->weight / ONEYEAR);
                        unvisited[p] = 0;
                    }
                    for (const auto &ret: node->outputs) {
                        if (memData.contains(ret.second.first)) {
                            memData[ret.second.first].insert(ret.first);
                        } else {
                            memData.insert(std::make_pair(ret.second.first, std::unordered_set<long>{ret.first}));
                        }
                    }
                }
            }
        }
        double min_cTime = s->horizon + 0.1;
        if (exec < G->nb_nodes) {
            for (int i = 0; i < nprocs; i++) {
                if (cTime[i] < min_cTime && scheduled[i] != -1) {
                    min_cTime = cTime[i];
                }
            }
            global_time = min_cTime;
        }
    } while (exec < G->nb_nodes && global_time < s->horizon);
    if (G->nb_nodes - exec > 0) {
        r->time = s->horizon * ONEYEAR / ONEHOUR;
    } else {
        r->time = (global_time - s->start) * ONEYEAR / ONEHOUR;
    }
    r->nb_faults = sF;
}



void outputEachResult(const std::vector<double> &res, const std::vector<int> &timeouts, const std::vector<double> &sfs,
                      const std::vector<std::string> &strategy,
                      const std::string &outfile,
                      const std::string &file,
                      const std::string &dag, const std::string &nbproc,
                      const std::string &failure_rate, const std::string &per_chains,
                      const std::string &bandwidth,
                      const long taketime,
                      double horizon,const std::vector<double>& time_token) {
    std::ofstream out(outfile, std::ios::app);
    for (int i = 0; i < res.size(); i++) {
        out << file << "," << dag << "," << nbproc << "," << failure_rate << "," << per_chains << "," << bandwidth <<
                "," << strategy[i]
                << "," << res[i] << "," << timeouts[i] << "," << sfs[i] << "," << taketime << "," << horizon <<
                    ","<<time_token[i]<<"\n";
    }
    out.close();
}



void initParser(int argc, char *argv[], param_t *runner) {
    int opt;
    const char *optstring = "f:s:d:o:i:c:v:w:p:t:m:";
    // -c checkpoint -v verification -s sample -l lambda -o outdir -i indir -d dependencedir -t strategy -c ckpt -v verif -m mode(1 wfgen 2 daggen)
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 's':
                runner->sample = std::stoi(optarg);
                break;
            case 'd':
                runner->dpddir=optarg;
                break;
            case 'o':
                runner->outdir = optarg;
                break;
            case 'i':
                runner->indir = optarg;
                break;
            case 'f':
                runner->pfails = split(optarg, ',');
                break;
            case 'c':
                runner->Cs = split(optarg, ',');
                break;
            case 'v':
                runner->Vs = split(optarg, ',');
                break;
            case 'w':
                runner->workflows = split(optarg, ',');
                break;
            case 'p':
                runner->P = split(optarg, ',');
                break;
            case 't':
                runner->strates = split(optarg, ',');
                for (auto &st: runner->strates) {
                    runner->strates_name.push_back(code2strategy[st]);
                }
                break;
            case 'm':
                runner->mode = std::stoi(optarg);
                break;
            default:
                std::cout << optstring << std::endl;
                std::cout << "Error in parsing arguments\n";
                break;
        }
    }
}

void simu_wfgen(const param_t &runner) {
    unsigned int type = runner.strates.size();
    for (const auto &p_fail: runner.pfails) {
        for (const auto &w: runner.workflows) {
            for (const auto &p: runner.P) {
                for (int i = 0; i < runner.Cs.size(); i++) {
                    time_t start = 0, end = 0;
                    time(&start);
                    std::vector<double> res = std::vector<double>(type, 0);
                    std::vector<int> timeouts = std::vector<int>(type, 0);
                    std::vector<double> sfs = std::vector<double>(type, 0.0);
                    std::vector<graph_t> test_graph = std::vector<graph_t>(type);
                    std::vector<double> time_token=std::vector<double>(type, 0.0);
                    int nprocs;
                    std::string filename = w;
                    filename.append("_").append(p).append("_").append(p_fail).append("_").append(runner.Cs[i]).append(
                        ".csv");
                    std::string dfile= runner.dpddir+w+".csv";
                    try {
                        for (int k = 0; k < type; k++) {
                            readGraph(&test_graph[k], runner.indir + runner.strates[k] + "_" + filename, &nprocs,dfile,std::stod(runner.Cs[i]));
                            initG(&test_graph[k]);
                        }
                    }catch (const std::exception &e) {
                        std::cout << e.what() << std::endl;
                        continue;
                    }
                    double lam = test_graph[0].lam;
                    double horizon = test_graph[0].makespan;
                    horizon /= ONEYEAR;
                    horizon *= 6;
#pragma omp parallel num_threads(PROC_CORES>30?30:PROC_CORES) shared(timeouts,res,sfs,time_token)
                    {
#pragma omp for reduction(+:timeouts,res,sfs,time_token)
                        for (int j = 0; j < runner.sample; j++) {
                            simulator_t simulator;
                            std::vector<graph_t> tg = test_graph;
                            initSimulator(nprocs, lam, horizon, &simulator);
                            for (int k = 0; k < type; k++) {
                                result_t result;
                                if (runner.strates[k] != "6") {
                                        time_t ss=0,ee=0;
                                        time(&ss);
                                        startSimulationSF(nprocs, &simulator, &result, &tg[k]);
                                        time(&ee);
                                        time_token[k]+=std::difftime(ee,ss);
                                }
                                else {
                                        time_t ss=0,ee=0;
                                        time(&ss);
                                        startMCSimulationNoCkpt(nprocs, &simulator, &result, &tg[k]);
                                        time(&ee);
                                        time_token[k]+=std::difftime(ee,ss);
                                }
                                if (result.time == horizon * ONEYEAR / ONEHOUR) {
                                    timeouts[k]++;
                                }
                                res[k] += std::log(result.time);
                                if (result.nb_faults > 0)
                                    sfs[k] += std::log(result.nb_faults);
                                for (auto &t: simulator.silent_errors) {
                                    t.initI();
                                }
                            }
                        }
                    }
                    std::cout << "RES: ";
                    for (auto &r: res) {
                        r = std::exp(r / runner.sample);
                        std::cout << r << " ";
                    }
                    for (auto &s: sfs) {
                        s = std::exp(s / runner.sample);
                    }
                    for(auto &t:time_token) {
                        t = t/runner.sample;
                    }
                    std::cout << "\n";
                    time(&end);
                    outputEachResult(res, timeouts, sfs, runner.strates_name, runner.outdir, filename,
                                                        w, p, p_fail, "0", runner.Cs[i],
                                                        end - start, horizon * ONEYEAR / ONEHOUR,time_token);
                }
            }
        }
    }
}

void simu_daggen(const param_t &runner) {
    //w number of nodes
    unsigned int type = runner.strates.size();
    std::string width[3] = {  "0.2","0.5","0.8"};
    std::string regularity[2] = { "0.2","0.8"};//
    std::string jump[4] = {"0","1", "2", "4"};
    std::string density[1] = {"0.2"};
    std::string chain[4] = {"0.01", "0.1", "0.2", "0.5"};
    for (const auto &p_fail: runner.pfails) {
        for (const auto &n: runner.workflows) {
            for (const auto &w: width) {
                for (const auto &rg: regularity) {
                    for (const auto &d: density) {
                        for (const auto &jp: jump) {
                            for (const auto &ch: chain) {
                                for (const auto &p: runner.P) {
                                    for (int i = 0; i < runner.Cs.size(); i++) {
                                        time_t start = 0, end = 0;
                                        time(&start);
                                        std::string dag = n;
                                        dag.append("_").append(w).append("_").append(rg).append("_").append(d).
                                                append("_").append(jp);
                                        std::string filename = dag;
                                        filename.append("_").append(ch).append("_").append(p).
                                                append("_").append(p_fail).append("_").append(runner.Cs[i]).append(
                                                    ".csv");
                                        std::string dfile=runner.dpddir;
                                        dfile.append(dag).append("_").append(ch).append(".csv");
                                        std::vector<double> res = std::vector<double>(type, 0);
                                        std::vector<int> timeouts = std::vector<int>(type, 0);
                                        std::vector<double> sfs = std::vector<double>(type, 0);
                                        std::vector<graph_t> test_graph = std::vector<graph_t>(type);
                                        std::vector<double> time_token=std::vector<double>(type, 0);
                                        int nprocs;
                                        try {
                                            for (int k = 0; k < type; k++) {
                                                readGraph(&test_graph[k], runner.indir + runner.strates[k] + "_" + filename,
                                                          &nprocs,dfile,stod(runner.Cs[i]));
                                                initG(&test_graph[k]);
                                            }
                                        }catch (const std::exception &e) {
                                            std::cout << e.what() << std::endl;
                                            continue;
                                        }
                                        double lam = test_graph[0].lam;
                                        double horizon = test_graph[0].makespan;
                                        horizon /= ONEYEAR;
                                        horizon *= 6;
#pragma omp parallel num_threads(PROC_CORES>30?30:PROC_CORES) shared(timeouts,res,sfs,time_token)
                                        {
#pragma omp for reduction(+:timeouts,res,sfs,time_token)
                                            for (int j = 0; j < runner.sample; j++) {
                                                simulator_t simulator;
                                                std::vector<graph_t> tg = test_graph;
                                                initSimulator(nprocs, lam, horizon, &simulator);
                                                for (int k = 0; k < type; k++) {
                                                    result_t result;
                                                    if (runner.strates[k] != "6") {
                                                            time_t ss=0,ee=0;
                                                            time(&ss);
                                                            startSimulationSF(nprocs, &simulator, &result, &tg[k]);
                                                            time(&ee);
                                                            time_token[k]+=std::difftime(ee,ss);

                                                    }
                                                    else {
                                                            time_t ss=0,ee=0;
                                                            time(&ss);
                                                            startMCSimulationNoCkpt(nprocs, &simulator, &result, &tg[k]);
                                                            time(&ee);
                                                            time_token[k]+=std::difftime(ee,ss);
                                                    }
                                                    if (result.time == horizon * ONEYEAR / ONEHOUR) {
                                                        timeouts[k]++;
                                                    }
                                                    res[k] += std::log(result.time);
                                                    if (result.nb_faults > 0)
                                                        sfs[k] += std::log(result.nb_faults);
                                                    for (auto &t: simulator.silent_errors) {
                                                        t.initI();
                                                    }
                                                }
                                            }
                                        }
                                        std::cout << filename<<" RES : ";
                                        for (auto &r: res) {
                                            r = std::exp(r / runner.sample);
                                            std::cout << r << " ";
                                        }
                                        for (auto &s: sfs) {
                                            s = std::exp(s / runner.sample);
                                        }
                                        for(auto &t: time_token) {
                                            t = t/runner.sample;
                                        }
                                        std::cout << "\n";

                                        time(&end);

                                        outputEachResult(res, timeouts, sfs, runner.strates_name, runner.outdir,
                                                         filename,
                                                         dag, p, p_fail, ch, runner.Cs[i], end - start,
                                                         horizon * ONEYEAR / ONEHOUR,time_token);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


int main(int argc, char *argv[]) {
    param_t runner;
    std::string outdir;
    initParser(argc, argv, &runner);
    switch (runner.mode) {
        case 1:
            simu_wfgen(runner);
            break;
        case 2:
            simu_daggen(runner);
            break;
        default:
            std::cerr << "Invalid Mode!\n";
            break;
    }
    return 0;
}

