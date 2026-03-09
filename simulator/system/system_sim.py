import os
import multiprocessing
from .simulator import Simulator
from .hardware_sim import HardwareSim
from .hardware import Hardware, H100
from .scheduler import Scheduler

def run_system_sim(model, trace,
                   hw_node_name, num_nodes, parallelism, io_algo, 
                   scheduler_algo, prefill_chunk=2048,
                   sim_method='roofline', # roofline, llmcompass
                   end_reqs=500, # should be set based on the request rate and workload
                   max_ctx_len = 8192*4,
                   workspace_dir = 'workspace/',
                   ):

    # This actually doesn't matter ??
    eval_hardware = Hardware(node=H100, # 硬件节点（flops / mem_bw / mem_size 等）
                             num_nodes=num_nodes,  # GPU 数量
                             parallelism=parallelism,  # (ep, tp, pp, cp)
                             io_algo=io_algo,  # 比如 'multishot'
    )

    # HardwareSim 是「给定一个 SimKernel，算出它要多少时间」的工具
    hardware_sim = HardwareSim(
        hardware=eval_hardware,
        method=sim_method, # 'roofline' or 'llmcompass'
        scheduler_algo=scheduler_algo,
        max_ctx_len = max_ctx_len, # 最大上下文长度，用于显存估算
    )

    # prefill_pool / decode_pool（当前已入池、等待被 batch 的任务），决定每一轮怎么决策
    scheduler = Scheduler(
        algo=scheduler_algo,
        prefill_chunk=prefill_chunk,
    )

    eval_hardware.node.name = hw_node_name

    sim = Simulator(
        model = model,
        trace=trace,
        scheduler=scheduler,
        hardware_sim=hardware_sim,
        end_time=500,
        start_reqs=0,
        end_reqs=end_reqs,
        workspace_dir=workspace_dir,
    )
    sim.run()
