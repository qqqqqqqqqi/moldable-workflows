#include <cstdlib>
#include <limits>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <omp.h>
#include <filesystem>
#include <getopt.h>

#define INF std::numeric_limits<double>::max();

#include "Trace.hpp"


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
    {"0", "ckptAll"}, {"1", "ckptCrossover"}, {"4", "our strategy"}, {"6", "ckptNone"}
};

struct param_t {
    int sample;
    std::string indir;
    std::string outdir;
    std::vector<std::string> Cs; //ckpt cost
    std::vector<std::string> Vs; //verif cost
    std::vector<std::string> pfails;
    std::vector<std::string> workflows;
    std::vector<std::string> P;
    std::vector<std::string> strategy;
    std::vector<std::string> strategy_name;
    std::vector<std::string> chain;
    int mode;
};

struct node_t {
    long id;
    double weight;
    std::string label;
    std::vector<std::pair<long, std::pair<std::string, double> > > inputs;
    std::vector<std::pair<long, std::pair<std::string, double> > > outputs;

    bool checkpoint;
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
    std::vector<node_t> nodes; //the two different possible weights for the tasks
    std::vector<std::vector<long> > schedule; //The list scheduling of the graph
    double makespan;
    double lam;
};

struct simulator_t {
    double start;
    double horizon; //the stop time
    std::vector<Trace> silent_errors;
    long global_nF; //number of fail-stop errors for the platform
    double muF; //mu of fail-stop error
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

    //Initialize the lists of faults for each processor
    s->global_nF = 0;
    for (int i = 0; i < nprocs; i++) {
        s->silent_errors.emplace_back(s->muF,  s->horizon);
        s->global_nF += s->silent_errors[i].nF;
    }
}


void readGraph(graph_t *G, std::string filename, int *n) {
    std::map<std::string, long> dict;
    std::ifstream input(filename, std::ios::in);
    if (!input.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(0);
        // return;
    }
    std::string line;
    std::getline(input, line);
    while (line.substr(0, 7) != "task_id")
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

    input.clear();
    input.seekg(0, std::ifstream::beg);
    std::getline(input, line);
    std::getline(input, line);
    while (line.substr(0, 7) != "task_id") {
        split_elems = split(line, ',');
        for (unsigned i = 2; i < split_elems.size(); i += 2) {
            G->nodes[dict[split_elems[0]]].outputs.emplace_back(dict[split_elems[1]],
                                                                std::pair<std::string, double>(
                                                                    split_elems[i], std::stod(split_elems[i + 1])));
            G->nodes[dict[split_elems[1]]].inputs.emplace_back(dict[split_elems[0]],
                                                               std::pair<std::string, double>(
                                                                   split_elems[i], std::stod(split_elems[i + 1])));
        }
        std::getline(input, line);
    }
    *n = p;
    input.close();
}

bool isReady(graph_t *G, long task) {
    for (const auto &p: G->nodes[task].inputs) {
        if (!G->nodes[p.first].done) {
            return false;
        }
    }
    return true;
}



void startSimulationNoCkpt(int nprocs, simulator_t *s, result_t *r, graph_t *G) {
    std::vector<double> cTime = std::vector(nprocs, s->start);
    std::vector<long> scheduled = std::vector<long>(nprocs, -1);
    std::vector<std::vector<long> > listScheduling = G->schedule;
    std::vector<int> currentTask = std::vector<int>(nprocs, -1);
    std::vector<bool> errorState = std::vector<bool>(nprocs, false);
    double global_time = s->start;
    int exec = 0;
    int sF = 0;
    do {
#ifdef DEBUG
        std::cout << "===Time : " << global_time << "===\n";
#endif
        /****************************
        BEGIN SCHEDULE
        ****************************/
        std::vector<bool> unvisited = std::vector<bool>(nprocs, true);
        for (int i = 0; i < nprocs; i++) {
            if (unvisited[i] && scheduled[i] != -1 && global_time >= cTime[i]) {
                node_t *node = &G->nodes[scheduled[i]];
                std::vector<int> processors = node->procs;
                node->done = true;
                exec++;
#ifdef DEBUG
                        std::cout << "task " << node->label << " done.\n";
                        std::cout << "Executed : " << exec << "\n";
#endif
                for (const auto &p: processors) {
                    unvisited[p] = false;
                }
            }
        }


        /****************************
             UPDATE READY TASKS
        ****************************/
        std::vector<long> nextTasks(nprocs, -1);
        for (int i = 0; i < nprocs; i++) {
            if (currentTask[i] < (int) listScheduling[i].size() - 1)
                nextTasks[i] = listScheduling[i][currentTask[i] + 1];
        }
        for (int i = 0; i < nprocs; i++) {
            if (cTime[i] <= global_time) {
                //If cTime[i]>global time : task on processor i is still running
                if (nextTasks[i] != -1) {
                    long next_task = nextTasks[i];
                    if (isReady(G, next_task)) {
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

        // std::vector<bool> hasChanged= std::vector<bool>(nprocs, false);
        for (int i = 0; i < nprocs; i++) {
            if (scheduled[i] != -1 && cTime[i] <= global_time) {
                // double nextError = s->silent_errors[i].next(global_time);
                double nextError = s->silent_errors[i].next(cTime[i]);
                double weight = G->nodes[scheduled[i]].weight;
                weight /= ONEYEAR;
                //silent error
                if (nextError < cTime[i] + weight) {
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
        std::fill(unvisited.begin(), unvisited.end(), true);
        for (int i = 0; i < nprocs; i++) {
            if (unvisited[i] && scheduled[i] != -1 && cTime[i] <= global_time) {
                node_t *node = &G->nodes[scheduled[i]];
                const std::vector<int> processors = node->procs;
                // no C / V
                for (const auto &p: processors) {
                    cTime[p] += node->weight / ONEYEAR;
                    unvisited[p] = false;
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

        /**
         *Verification
         */
        if (exec == G->nb_nodes && global_time < s->horizon) {
            bool rollback = false;
            double V = 0;
            double C = 0;
            for (auto &sink: G->nodes) {
                if (sink.V != 0) {
                    V += sink.V;
                    C += sink.C;
                }
            }
            for (int i = 0; i < nprocs; i++) {
                if (errorState[i]) {
                    rollback = true;
                    break;
                }
            }
            if (rollback) {
                exec = 0;
                for (int i = 0; i < nprocs; i++) {
                    currentTask[i] = -1;
                    scheduled[i] = -1;
                    errorState[i] = false;
                    cTime[i] = global_time + V / ONEYEAR;
                }
                for (auto node: G->nodes) {
                    node.done = false;
                }
            } else {
                for (int i = 0; i < nprocs; i++) {
                    cTime[i] += global_time + V / ONEYEAR + C / ONEYEAR;
                }
            }
        }
    } while (exec < G->nb_nodes && global_time < s->horizon);

    if (G->nb_nodes - exec > 0) {
        r->time = s->horizon * ONEYEAR / ONEHOUR;
    } else {
        r->time = (global_time - s->start) * ONEYEAR / ONEHOUR;
    }
    r->nb_faults = sF;
}

void startSimulationSF(int nprocs, simulator_t *s, result_t *r, graph_t *G) {
    std::vector<double> cTime = std::vector(nprocs, s->start);
    std::vector<long> scheduled = std::vector<long>(nprocs, -1);
    std::vector<std::vector<long> > listScheduling = G->schedule;
    std::vector<int> currentTask = std::vector<int>(nprocs, -1);
    //lastCkpt[p] will contain the index in listScheduling[p] of the last checkpointed task for processor p
    std::vector<int> lastCkpt = std::vector<int>(nprocs, -1);
    std::vector<bool> errorState = std::vector<bool>(nprocs, false);

    double global_time = s->start;
    int exec = 0;
    int sF = 0;
    std::map<std::string, std::set<long> > memData; //file:<outputs>
    do {
#ifdef DEBUG
        std::cout << "===Time : " << global_time << "===\n";
#endif
        /****************************
        BEGIN SCHEDULE
        ****************************/
        std::vector<bool> unvisited = std::vector<bool>(nprocs, true);
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
#ifdef DEBUG
                                    std::cout << G->nodes[listScheduling[p][k]].label << " rollback need to re-execute\n";
#endif
                            }
                        }
                        errorState[p] = false;
                        currentTask[p] = lastCkpt[p]; //NEED TO ROLLBACK
                        unvisited[p] = false;
                    }
                    node->failed = false;
                } else {
                    node->done = true;
                    exec++;
#ifdef DEBUG
                        std::cout << "task " << node->label << " done.\n";
                        std::cout << "Executed : " << exec << "\n";
#endif
                    for (const auto &p: processors) {
                        unvisited[p] = false;
                    }
                }
            }
        }

        /****************************
             UPDATE READY TASKS
        ****************************/
        std::vector<long> nextTasks(nprocs, -1);
        for (int i = 0; i < nprocs; i++) {
            if (currentTask[i] < (int) listScheduling[i].size() - 1)
                nextTasks[i] = listScheduling[i][currentTask[i] + 1];
        }
        for (int i = 0; i < nprocs; i++) {
            if (cTime[i] <= global_time) {
                //If cTime[i]>global time : task on processor i is still running
                if (nextTasks[i] != -1) {
                    long next_task = nextTasks[i];
                    if (isReady(G, next_task)) {
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

        std::map<node_t *, double> rv;
        for (int i = 0; i < nprocs; i++) {
            if (scheduled[i] != -1 && cTime[i] <= global_time) {
                /** READ **/
                node_t *node = &G->nodes[scheduled[i]];
                double R = 0;
                if (!rv.contains(node)) {
                    //read from memory or read from stable storage
                    for (const auto &ret: node->inputs) {
                        if (!memData.contains(ret.second.first)) {
                            R += ret.second.second;
                        } else {
                            memData[ret.second.first].erase(node->id);
                            if (memData[ret.second.first].empty()) {
                                //if all successors read the file ,delete!
                                memData.erase(ret.second.first);
                            }
                        }
                    }
                    rv[node] = R;
                } else {
                    R = rv[node];
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
        *VERIFICATION / CHECKPOINT
        **/
        std::fill(unvisited.begin(), unvisited.end(), true);
        for (int i = 0; i < nprocs; i++) {
            if (unvisited[i] && scheduled[i] != -1 && cTime[i] <= global_time) {
                node_t *node = &G->nodes[scheduled[i]];
                const std::vector<int> processors = node->procs;
                double R = rv[node];
                if (node->checkpoint) {
                    bool rollBack = false;
                    for (const auto &p: processors) {
                        //Verification if need to rollback
                        cTime[p] += (R / ONEYEAR + node->weight / ONEYEAR + node->V / ONEYEAR);
                        unvisited[p] = false;
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
                                memData.insert(std::make_pair(ret.second.first, std::set<long>{ret.first}));
                            }
                        }
                    }
                } else {
                    // no C / V
                    for (const auto &p: processors) {
                        cTime[p] += (R / ONEYEAR + node->weight / ONEYEAR);
                        unvisited[p] = false;
                    }
                    for (const auto &ret: node->outputs) {
                        if (memData.contains(ret.second.first)) {
                            memData[ret.second.first].insert(ret.first);
                        } else {
                            memData.insert(std::make_pair(ret.second.first, std::set<long>{ret.first}));
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


void init() {
    struct timeval time;
    gettimeofday(&time, NULL);
    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
}


void outputEachResult(const std::vector<double> &res, const std::vector<int> &timeouts, const std::vector<double> &sfs,
                      const std::vector<std::string> &strategy,
                      const std::string &outfile,
                      const std::string &file,
                      const std::string &dag, const std::string &nbproc,
                      const std::string &failure_rate, const std::string &per_chains,
                      const std::string &bandwidth,
                      const long taketime,
                      double horizon) {
    std::ofstream out(outfile, std::ios::app);
    for (int i = 0; i < res.size(); i++) {
        out << file << "," << dag << "," << nbproc << "," << failure_rate << "," << per_chains << "," << bandwidth <<
                "," << strategy[i]
                << "," << res[i] << "," << timeouts[i] << "," << sfs[i] << "," << taketime << "," << horizon << "\n";
    }
    out.close();
}


void initParser(int argc, char *argv[], param_t *runner) {
    int opt;
    const char *optstring = "f:s:o:i:c:v:w:p:t:m:";
    // -c checkpoint  -s sample -l lambda -o outdir -i indir -t strategy  -m mode(1 wfgen 2 daggen)
    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 's':
                runner->sample = std::stoi(optarg);
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
            case 'w':
                runner->workflows = split(optarg, ',');
                break;
            case 'p':
                runner->P = split(optarg, ',');
                break;
            case 't':
                runner->strategy = split(optarg, ',');
                for (auto &st: runner->strategy) {
                    runner->strategy_name.push_back(code2strategy[st]);
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
    unsigned int type = runner.strategy.size();
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
                    int nprocs;
                    std::string filename = w;
                    filename.append("_").append(p).append("_").append(p_fail).append("_").append(runner.Cs[i]).append(
                        ".csv");
                    for (int k = 0; k < type; k++) {
                        readGraph(&test_graph[k], runner.indir + runner.strategy[k] + "_" + filename, &nprocs);
                    }
                    double lam = test_graph[0].lam;
                    double horizon = test_graph[0].makespan;
                    horizon /= ONEYEAR;
                    horizon *= 6;
#pragma omp parallel num_threads(PROC_CORES>30?30:1) shared(timeouts,res,sfs)
                    {
#pragma omp for reduction(+:timeouts,res,sfs)
                        for (int j = 0; j < runner.sample; j++) {
                            simulator_t simulator;
                            std::vector<graph_t> tg = test_graph;
                            initSimulator(nprocs, lam, horizon, &simulator);
                            for (int k = 0; k < type; k++) {
                                result_t result;
                                if (runner.strategy[k] != "6") {
                                    startSimulationSF(nprocs, &simulator, &result, &tg[k]);
                                }
                                else
                                    startSimulationNoCkpt(nprocs, &simulator, &result, &tg[k]);
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
                    std::cout << "\n";
                    time(&end);
                    outputEachResult(res, timeouts, sfs, runner.strategy_name, runner.outdir, filename,
                    w, p, p_fail, "0", runner.Cs[i], end - start, horizon * ONEYEAR / ONEHOUR);
                }
            }
        }
    }
}

void simu_daggen(const param_t &runner) {
    //w number of nodes
    unsigned int type = runner.strategy.size();
    std::string width[3] = {"0.2", "0.5", "0.8"};
    std::string regularity[2] = {"0.2", "0.8"};
    std::string jump[3] = {"1", "2", "4"};
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
                                        std::vector<double> res = std::vector<double>(type, 0);
                                        std::vector<int> timeouts = std::vector<int>(type, 0);
                                        std::vector<double> sfs = std::vector<double>(type, 0);
                                        std::vector<graph_t> test_graph = std::vector<graph_t>(type);
                                        int nprocs;
                                        for (int k = 0; k < type; k++) {
                                            readGraph(&test_graph[k], runner.indir + runner.strategy[k] + "_" + filename,
                                                      &nprocs);
                                        }
                                        double lam = test_graph[0].lam;
                                        double horizon = test_graph[0].makespan;
                                        horizon /= ONEYEAR;
                                        horizon *= 6;
#pragma omp parallel num_threads(PROC_CORES>30?30:PROC_CORES) shared(timeouts,res,sfs)

                                        {
#pragma omp for reduction(+:timeouts,res,sfs)
                                            for (int j = 0; j < runner.sample; j++) {
                                                simulator_t simulator;
                                                std::vector<graph_t> tg = test_graph;
                                                initSimulator(nprocs, lam, horizon, &simulator);
                                                for (int k = 0; k < type; k++) {
                                                    result_t result;
                                                    if (runner.strategy[k] != "6") {
                                                        startSimulationSF(nprocs, &simulator, &result, &tg[k]);
                                                    }
                                                    else {
                                                        startSimulationNoCkpt(nprocs, &simulator, &result, &tg[k]);
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
                                        std::cout << "RES : ";
                                        for (auto &r: res) {
                                            r = std::exp(r / runner.sample);
                                            std::cout << r << " ";
                                        }
                                        for (auto &s: sfs) {
                                            s = std::exp(s / runner.sample);
                                        }
                                        std::cout << "\n";
                                        time(&end);
                                        outputEachResult(res, timeouts, sfs, runner.strategy_name, runner.outdir,
                                                         filename,
                                                         dag, p, p_fail, ch, runner.Cs[i], end - start,
                                                         horizon * ONEYEAR / ONEHOUR);
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
