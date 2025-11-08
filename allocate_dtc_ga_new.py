import yaml
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import time
import random

import sys
import os
# 将上级目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AcAdjoint import AcAdjointFunction # GA 适应度评估只需要 forward，可以继续用
from AcSimulation import AcSimulationPardiso, AcSimulationCuDSS # 或 AcSimulationScipy
from Circuit import Circuit, BranchType
from build_ckt import build_ac_ckt



def get_z_mask_flat_target(frequencies_hz, flat_target_value_ohm):
    """
    根据一个固定的目标阻抗值生成平坦的 Z_mask。
    """
    return np.full_like(frequencies_hz, flat_target_value_ohm, dtype=float)

def create_individual(num_total_candidates, num_to_select):
    """创建一个随机个体 (染色体) - 包含 num_to_select 个唯一索引"""
    if num_to_select > num_total_candidates:
        raise ValueError(f"Cannot select {num_to_select} items from {num_total_candidates} candidates.")
    return random.sample(range(num_total_candidates), num_to_select)

def initialize_population(pop_size, num_total_candidates, num_to_select):
    """初始化种群"""
    return [create_individual(num_total_candidates, num_to_select) for _ in range(pop_size)]

def calculate_fitness(individual, sim_data_dict):
    """计算个体的适应度。sim_data_dict 现在是包含所有仿真所需数据的字典。"""
    frequencies = sim_data_dict["frequencies"]
    z_mask = sim_data_dict["z_mask_values"]
    sims = sim_data_dict["sims"]
    num_freq = len(frequencies)
    impedances = np.zeros(num_freq, dtype=float)

    # 初始化电容值为0
    c_value_current = torch.zeros_like(sim_data_dict["base_c_values_all"])
    # 被选中的染色体 初始化电容值为选中染色体的电容值
    if individual: # 如果个体不为空 (即有选中的电容)
        selected_indices_tensor = torch.tensor(individual, dtype=torch.long)
        c_value_current[selected_indices_tensor] = sim_data_dict["base_c_values_all"][selected_indices_tensor]

    # zs = []
    for i, (freq, sim_instance) in enumerate(zip(frequencies, sims)):
        i_voltage, _ = AcAdjointFunction.apply(
            sim_data_dict["g_index"], 
            sim_data_dict["r_index"],
            sim_data_dict["c_indices_all"],
            sim_data_dict["l_index"], 
            sim_data_dict["xc_index"], 
            sim_data_dict["xl_index"],
            sim_data_dict["i_index"], 
            sim_data_dict["v_index"],
            sim_data_dict["all_exc_index"],
            sim_data_dict["g_value"], 
            sim_data_dict["r_value"],
            c_value_current,
            sim_data_dict["l_value"], 
            sim_data_dict["xc_value"], 
            sim_data_dict["xl_value"],
            sim_data_dict["all_exc_value_zin_calc"],
            freq,
            sim_instance
        )
        # zs.append(i_voltage)
        impedances[i] = i_voltage.abs().item()

    # if not zs:
    #     impedances = torch.tensor([]) # 创建一个空的 PyTorch tensor
    # else:
    #     zs_tensor = torch.cat(zs)
    #     impedances = zs_tensor.abs() # impedances 现在是一个 PyTorch Tensor
    #     impedances_np = impedances.detach().cpu().numpy()

    if np.isinf(impedances).any() or np.isnan(impedances).any():
        n_no_ok = float('inf')
    else:
        violation_points = impedances >= z_mask # Tensor vs Tensor 比较
        n_no_ok = np.sum(violation_points)
        # n_no_ok = torch.sum(violation_points).item()    # torch.sum 返回 Tensor，.item() 转为 Python 数字
    
    # 计算 RMSE 公式3
    rmse = 0.0  # 默认值为 0.0，表示没有违规
    if n_no_ok > 0:
        zin_viol = impedances[violation_points]
        zmask_viol = z_mask[violation_points]
        relative_errors_sq = ((zin_viol - zmask_viol) / zin_viol) ** 2
        rmse = np.sqrt(np.mean(relative_errors_sq))
    
    # 成本函数 满足论文中公式2的点越多，rank越低
    fitness = 1.0 / (1.0 + n_no_ok + rmse * 10)
    return fitness, n_no_ok, rmse

# 轮盘赌算法
def roulette_wheel_selection(population, fitness_scores, num_to_select):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        # 为0则随机选
        return random.choices(population, k=num_to_select)
    # 根据适应度设置个体被选中的概率
    selection_probs = [f / total_fitness for f in fitness_scores]
    prob_sum = sum(selection_probs)
    if prob_sum == 0:
       return random.choices(population, k=num_to_select)
    selection_probs = [p / prob_sum for p in selection_probs]
    selected_indices = np.random.choice(len(population), size=num_to_select, p=selection_probs, replace=True)
    # 从原始种群中提取出被选中的个体
    return [population[i] for i in selected_indices]

# 从父代基因中交叉产生子代
def single_point_crossover(parent1, parent2, num_decaps_to_select, num_total_candidates):
    if not parent1 or not parent2 or num_decaps_to_select == 0 : # Cannot crossover empty or single element
         return parent1[:], parent2[:]
    if len(parent1) != num_decaps_to_select or len(parent2) != num_decaps_to_select:
         # This can happen if parents are from elites and were shorter than num_decaps_to_select (e.g. if num_decaps_to_select=0)
         # Or if initialization didn't strictly enforce length.
         # Fallback: no crossover or handle appropriately. For now, no crossover if lengths differ.
         return parent1[:], parent2[:]

    # --- 在进行任何选择前，先打乱父代基因的副本 后续交叉产生子代 ---
    parent1_shuffled = random.sample(parent1, len(parent1))
    parent2_shuffled = random.sample(parent2, len(parent2))
    point = random.randint(1, num_decaps_to_select - 1) if num_decaps_to_select > 1 else 0

    # 从parent1继承point个，再从parent2中继承num_decaps_to_select - point个组成新的子代
    child1_set = set(parent1_shuffled[:point])
    for gene in parent2_shuffled:
        if len(child1_set) < num_decaps_to_select:
            child1_set.add(gene)
        else:
            break
    # Fill if not enough unique genes
    while len(child1_set) < num_decaps_to_select:
        candidate_gene = random.randint(0, num_total_candidates - 1)
        child1_set.add(candidate_gene)
    child1 = list(child1_set)
    random.shuffle(child1) # Maintain list representation but order doesn't strictly matter for sets of indices

    child2_set = set(parent2_shuffled[:point])
    for gene in parent1_shuffled:
        if len(child2_set) < num_decaps_to_select:
            child2_set.add(gene)
        else:
            break
    while len(child2_set) < num_decaps_to_select:
        candidate_gene = random.randint(0, num_total_candidates - 1)
        child2_set.add(candidate_gene)
    child2 = list(child2_set)
    random.shuffle(child2)

    return child1[:num_decaps_to_select], child2[:num_decaps_to_select]


# 突变
def mutate(individual, num_total_candidates, mutation_rate, num_decaps_to_select):
    if not individual or num_decaps_to_select == 0: return individual # Cannot mutate empty
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # Replace with a new unique candidate index
            current_gene = mutated_individual[i]
            new_gene = current_gene
            attempts = 0
            while new_gene == current_gene or new_gene in mutated_individual: # Ensure new and not already present
                new_gene = random.randint(0, num_total_candidates - 1)
                attempts += 1
                if attempts > num_total_candidates * 2: # Avoid infinite loop if space is tight
                    break # Give up on this mutation
            if new_gene != current_gene and new_gene not in mutated_individual:
                 mutated_individual[i] = new_gene
    return mutated_individual

# --- Plotting Function ---
def plot_zin_curves(individual_to_plot, 
                    sim_data_dict, 
                    plot_title_prefix, 
                    case_name, 
                    num_decaps, 
                    output_filename):
    """
    Generates and saves a log-log plot of Z_in vs frequency.
    Includes: Z_in for the given individual, Z_mask, and baseline Z_in (no decaps).
    """
    print(f"Plotting for: {plot_title_prefix}, Nd={num_decaps}")
    frequencies = sim_data_dict["frequencies"]
    z_mask = sim_data_dict["z_mask_values"]
    sims = sim_data_dict["sims"] # List of pre-initialized simulators for each frequency

    plt.figure(figsize=(12, 7))

    # 1. Calculate and Plot Z_in for the given individual
    zin_optimized = np.zeros_like(frequencies)
    c_val_optimized = torch.zeros_like(sim_data_dict["base_c_values_all"])
    if individual_to_plot: # Check if list is not empty
        selected_indices = torch.tensor(individual_to_plot, dtype=torch.long)
        if selected_indices.numel() > 0: # Check if tensor is not empty
            if selected_indices.max() < sim_data_dict["base_c_values_all"].shape[0]:
                 c_val_optimized[selected_indices] = sim_data_dict["base_c_values_all"][selected_indices]
            else:
                print(f"Error: Invalid index in individual_to_plot for plotting. Max index: {selected_indices.max()}, Base C shape: {sim_data_dict['base_c_values_all'].shape[0]}")
                # Handle error or skip plotting this curve
    
    for i, (freq, sim_instance) in enumerate(zip(frequencies, sims)):
        try:
            i_voltage, _ = AcAdjointFunction.apply(
                sim_data_dict["g_index"], sim_data_dict["r_index"], sim_data_dict["c_indices_all"],
                sim_data_dict["l_index"], sim_data_dict["xc_index"], sim_data_dict["xl_index"],
                sim_data_dict["i_index"], sim_data_dict["v_index"], sim_data_dict["all_exc_index"],
                sim_data_dict["g_value"], sim_data_dict["r_value"], c_val_optimized,
                sim_data_dict["l_value"], sim_data_dict["xc_value"], sim_data_dict["xl_value"],
                sim_data_dict["all_exc_value_zin_calc"], freq, sim_instance
            )
            zin_optimized[i] = i_voltage.abs().item()
        except Exception as e:
            print(f"Error during optimized Zin simulation for plotting at freq {freq}: {e}")
            zin_optimized[i] = np.nan
    
    plt.loglog(frequencies, zin_optimized, label=f'Optimized $Z_{{in}}$ (Nd={num_decaps})', linewidth=2)

    # 2. Plot Z_mask
    plt.loglog(frequencies, z_mask, '--', label='$Z_{{mask}}$ Target', color='red')

    # 3. Calculate and Plot Baseline Z_in (No Decaps)
    zin_baseline = np.zeros_like(frequencies)
    c_val_baseline = torch.zeros_like(sim_data_dict["base_c_values_all"]) # All zeros
    for i, (freq, sim_instance) in enumerate(zip(frequencies, sims)):
        try:
            i_voltage, _ = AcAdjointFunction.apply(
                sim_data_dict["g_index"], sim_data_dict["r_index"], sim_data_dict["c_indices_all"],
                sim_data_dict["l_index"], sim_data_dict["xc_index"], sim_data_dict["xl_index"],
                sim_data_dict["i_index"], sim_data_dict["v_index"], sim_data_dict["all_exc_index"],
                sim_data_dict["g_value"], sim_data_dict["r_value"], c_val_baseline, # Use baseline C values
                sim_data_dict["l_value"], sim_data_dict["xc_value"], sim_data_dict["xl_value"],
                sim_data_dict["all_exc_value_zin_calc"], freq, sim_instance
            )
            zin_baseline[i] = i_voltage.abs().item()
        except Exception as e:
            print(f"Error during baseline Zin simulation for plotting at freq {freq}: {e}")
            zin_baseline[i] = np.nan

    plt.loglog(frequencies, zin_baseline, ':', label='$Z_{{in}}$ (No Decaps)', color='gray')

    # Plot details
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Impedance Magnitude (Ohm)")
    plt.title(f"{plot_title_prefix} - PDN Impedance ({case_name}, Nd={num_decaps})")
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.legend()

    # Dynamic Y-axis limits
    upper_y_limit = 1.0 # Default
    all_valid_data = np.concatenate([
        zin_optimized[~np.isnan(zin_optimized) & ~np.isinf(zin_optimized)],
        zin_baseline[~np.isnan(zin_baseline) & ~np.isinf(zin_baseline)],
        z_mask[~np.isnan(z_mask) & ~np.isinf(z_mask)]
    ])
    if all_valid_data.size > 0:
        upper_y_limit = max(upper_y_limit, np.max(all_valid_data) * 1.5) # Add some headroom
    
    min_y_limit = 1e-4 # Default bottom
    if all_valid_data.size > 0:
        min_y_limit = min(1e-3, np.min(all_valid_data) * 0.5 )
        min_y_limit = max(1e-4, min_y_limit) # ensure it's not too low

    plt.ylim(bottom=min_y_limit, top=upper_y_limit)
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
    plt.close()

# 新函数：为固定的 Nd 运行 GA
def run_ga_for_nd(num_decaps_to_select_param,
                  case_name_param,
                  ga_params,                 
                  sim_data_dict_base,
                  num_total_candidates,      
                  plot_intermediate=False, 
                  plot_final_for_this_nd=False,
                  output_plot_dir="plots" # Directory for plots
                  ):
    """为固定的电容数量(Nd)运行一次GA，返回最佳个体和其N_no_ok。"""

    POPULATION_SIZE = ga_params["POPULATION_SIZE"]
    NUM_GENERATIONS = ga_params["NUM_GENERATIONS"]
    CROSSOVER_RATE = ga_params["CROSSOVER_RATE"]
    MUTATION_RATE = ga_params["MUTATION_RATE"]
    paper_X_rate = ga_params.get("paper_X_rate", 0.5) # 从参数获取，默认0.5

    current_num_decaps_to_select = num_decaps_to_select_param
    sim_data_dict_local = sim_data_dict_base # 直接使用传入的字典

    # --- 处理 Nd=0 的特殊情况 ---
    if current_num_decaps_to_select == 0:
        print(f"  [Nd=0] Calculating baseline fitness.")
        fitness, n_no_ok, rmse = calculate_fitness([], sim_data_dict_local)
        print(f"  [Nd=0] Result: Fitness={fitness:.4f}, N_no_ok={n_no_ok}, rmse={rmse}")
        if plot_final_for_this_nd:
             # plot_zin_curve(...) # 可以选择绘制基线图
            os.makedirs(output_plot_dir, exist_ok=True)
            plot_filename = os.path.join(output_plot_dir, f"{case_name_param}_Zin_Nd0_Baseline.png")
            plot_zin_curves([], sim_data_dict_local, "Baseline (Nd=0)", case_name_param, 0, plot_filename)

        return [], n_no_ok # rmses 返回空个体和对应的 n_no_ok 

    # --- 处理 Nd > num_total_candidates ---
    if num_total_candidates < current_num_decaps_to_select:
         print(f"  [Nd={num_decaps_to_select_param}] Warning: Cannot select {current_num_decaps_to_select} from {num_total_candidates}. Skipping run.")
         return None, float('inf') # 返回无效结果

    # --- 初始化 GA ---
    start_time_inner = time.time()
    population = initialize_population(POPULATION_SIZE, num_total_candidates, current_num_decaps_to_select)
    best_individual_inner = None
    best_fitness_inner = -1.0
    best_n_no_ok_inner = float('inf')
    # fitness_history_inner = []

    # --- GA 循环 (固定迭代次数，例如40代) ---
    for generation in range(NUM_GENERATIONS):
        gen_start_time_inner = time.time()
        fitness_scores = []
        n_no_ok_scores = []
        rmses = []
        for i, individual in enumerate(population):
            fitness, n_no_ok_val, rmse = calculate_fitness(individual, sim_data_dict_local)
            fitness_scores.append(fitness)
            n_no_ok_scores.append(n_no_ok_val)
            rmses.append(rmse)

        # --- 追踪本轮 GA 的最佳解 ---
        current_best_gen_idx = np.argmax(fitness_scores)
        current_best_gen_fitness = fitness_scores[current_best_gen_idx]
        current_best_gen_n_no_ok = n_no_ok_scores[current_best_gen_idx]
        # fitness_history_inner.append(current_best_gen_fitness)

        if current_best_gen_fitness > best_fitness_inner:
            best_fitness_inner = current_best_gen_fitness
            best_individual_inner = population[current_best_gen_idx][:]
            best_n_no_ok_inner = current_best_gen_n_no_ok
        
        # --- 打印简化的进度 (避免过多输出) ---
        # if (generation + 1) % 10 == 0 or generation == NUM_GENERATIONS - 1 : # 每10代打印一次
        avg_fitness = np.mean(fitness_scores)
        print(f"    [Nd={current_num_decaps_to_select}] Gen {generation+1:02d}/{NUM_GENERATIONS} - "
            f"Best(run): {best_fitness_inner:.4f} ({best_n_no_ok_inner}), "
            f"GenBest: {current_best_gen_fitness:.4f} ({current_best_gen_n_no_ok}), "
            f"Avg: {avg_fitness:.4f}, "
            f"Time: {time.time() - gen_start_time_inner:.2f}s")

        # --- 检查是否在本轮GA内部就找到了完美解 ---
        if best_n_no_ok_inner == 0:
            print(f"  [Nd={current_num_decaps_to_select}] Perfect solution found at Gen {generation+1}. Stopping early for this Nd.")
            break # 提前结束本轮 Nd 的 GA
        elif best_n_no_ok_inner >= 10:
            print(f"相差太多，提前结束")
            break

        # --- 生成下一代 ---
        next_population = []
        # Elitism - 使用 paper_X_rate
        num_elites_to_keep = int(round(len(fitness_scores) * (1.0 - paper_X_rate))) 
        elite_indices = np.argsort(fitness_scores)[-num_elites_to_keep:]
        for idx in elite_indices:
             next_population.append(population[idx][:])

        # Offspring Generation
        num_offspring_needed = POPULATION_SIZE - num_elites_to_keep
        num_parents_to_select = num_offspring_needed if num_offspring_needed % 2 == 0 else num_offspring_needed + 1
        if num_parents_to_select > 0 :
            parents = roulette_wheel_selection(population, fitness_scores, num_parents_to_select)
            for i in range(0, num_offspring_needed, 2):
                 if i+1 < len(parents): parent1, parent2 = parents[i], parents[i+1]
                 else: parent1, parent2 = random.choice(parents), random.choice(parents)
                 
                 if random.random() < CROSSOVER_RATE:
                     # 使用你修改后的交叉算子
                     child1, child2 = single_point_crossover(parent1, parent2, current_num_decaps_to_select, num_total_candidates) 
                 else: child1, child2 = parent1[:], parent2[:]
                 
                 child1 = mutate(child1, num_total_candidates, MUTATION_RATE, current_num_decaps_to_select)
                 child2 = mutate(child2, num_total_candidates, MUTATION_RATE, current_num_decaps_to_select)
                 
                 next_population.append(child1)
                 if len(next_population) < POPULATION_SIZE: next_population.append(child2)

        while len(next_population) < POPULATION_SIZE:
             next_population.append(create_individual(num_total_candidates, current_num_decaps_to_select))
        population = next_population[:POPULATION_SIZE]

    # --- GA Loop for current Nd Ends ---
    total_time_inner = time.time() - start_time_inner
    print(f"  [Nd={current_num_decaps_to_select}] GA run finished in {total_time_inner:.2f}s. Best N_no_ok found: {best_n_no_ok_inner}")

    # 可选的绘图
    if plot_final_for_this_nd and best_individual_inner is not None:
        # 需要一个绘图函数，这里假设我们有 plot_zin_curve_v2
        # plot_zin_curve_v2(best_individual_inner, sim_data_dict_local, f"Final_Nd{current_num_decaps_to_select}", case_name_param, current_num_decaps_to_select, output_suffix=f"run_Nd{current_num_decaps_to_select}")
        os.makedirs(output_plot_dir, exist_ok=True)
        plot_filename = os.path.join(output_plot_dir, f"{case_name_param}_Zin_Nd{current_num_decaps_to_select}_RunBest.png")
        plot_zin_curves(best_individual_inner, sim_data_dict_local, 
                        f"Run Best (Nd={current_num_decaps_to_select})", 
                        case_name_param, current_num_decaps_to_select, plot_filename)


    # 返回本轮GA找到的最佳个体及其 N_no_ok
    return best_individual_inner, best_n_no_ok_inner
def find_optimal_nd_iteratively():
    # --- 主要配置 (可以移到这里，或者作为参数传入) ---
    CASE_NAME = "ascend910"  # 可选: "case1", "micro150", "ascend910" "multigpu"
    NUM_FREQUENCY_POINTS = 100
    # 定义 GA 参数字典
    ga_run_params = {
        "POPULATION_SIZE": 50,
        "NUM_GENERATIONS": 40, # 每次 Nd 运行的固定代数
        "CROSSOVER_RATE": 0.9,
        "MUTATION_RATE": 0.1,
        "paper_X_rate": 0.5 # 使用 X_rate 控制精英比例
    }

    # --- 读取文件 ---
    file_design = f"data/{CASE_NAME}.yaml"
    file_prev_tsv_results = f"data/2025_11/{CASE_NAME}_result_tsv.yaml"
    
    print(f"Loading design: {file_design}")
    with open(file_design, "r") as f:
        design = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(f"Loading previous TSV results: {file_prev_tsv_results}")
    with open(file_prev_tsv_results, "r") as f:
        result_tsv_data = yaml.load(f.read(), Loader=yaml.FullLoader)

    interposer_w = design["interposer"]["w"] * 5
    interposer_h = design["interposer"]["h"] * 5
    vdd = design["vdd"]

    print("Building AC circuit model...")
    # 构建电路
    ckt = build_ac_ckt(file_design, file_prev_tsv_results)

    observe_branch = ["id"]
    candidate_coords = []
    for x in range(interposer_w):
        for y in range(interposer_h):
            if (x,y) not in result_tsv_data.get("tsvs", []):
                candidate_coords.append((x,y))
    candidate_dtc_names = [f"dtc_{x}_{y}" for x, y in candidate_coords] # 需要这个列表用于最后输出
    num_total_candidates = len(candidate_dtc_names)
    candidate_branch_names = [f"cd_{x}_{y}" for x, y in candidate_coords] # build_ac_ckt 可能需要这个？确认一下



    frequency_points = np.geomspace(0.1e9, 10e9, NUM_FREQUENCY_POINTS)
        
    # --- Z_mask (根据需要选择样式) ---
    # 目标阻抗，直接算的值
    chiplet_power = design["chiplets"][0]["power"]
    # target_ohm_value = 0.1 * vdd * vdd / chiplet_power
    target_impedance = 9999999
    for chiplet in design["chiplets"]:
        target_impedance = min(target_impedance, 0.1 * vdd * vdd / chiplet['power'])
    z_mask_target_values = get_z_mask_flat_target(frequency_points, target_impedance)
    print(f"Using flat target impedance: {target_impedance:.4f} Ohm for iterative search.")

    # --- 准备基础仿真数据字典 (只做一次) ---
    print("Preparing base simulation data...")
    all_branches_for_prep = observe_branch + candidate_branch_names
    typ, u, v, val, index_map = ckt.prepare_sim("0", all_branches_for_prep)

    c_indices_all_candidates = index_map[1:]
    base_c_values_all_candidates = val[c_indices_all_candidates]
    zin_port_index_in_typ = index_map[0]
    if typ[zin_port_index_in_typ] != BranchType.I: raise ValueError("Obs branch not Type I")
    (all_exc_indices_in_typ,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
    all_exc_base_values = val[all_exc_indices_in_typ]
    # 1. 创建一个全零的激励值张量
    all_exc_val_for_zin_calc = torch.zeros_like(all_exc_base_values, dtype=torch.complex128)
    is_zin_port_in_all_exc = (all_exc_indices_in_typ == zin_port_index_in_typ)
    if not torch.any(is_zin_port_in_all_exc): raise ValueError("Zin port not in excitation sources")
    all_exc_val_for_zin_calc[is_zin_port_in_all_exc] = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)


    print("Initializing simulators for each frequency (once)...")
    sim_init_start_time = time.time()
    sims = []
    for freq in frequency_points:
        sims.append(AcSimulationCuDSS(typ, u, v, val, freq))
    print(f"Simulators initialized in {time.time() - sim_init_start_time:.2f}s")

    sim_data_dict_base = {
        "frequencies": frequency_points,
        "z_mask_values": z_mask_target_values,
        "c_indices_all": c_indices_all_candidates,
        "base_c_values_all": base_c_values_all_candidates,
        "i_index": torch.tensor([zin_port_index_in_typ], dtype=torch.long),
        "all_exc_index": all_exc_indices_in_typ,
        "all_exc_value_zin_calc": all_exc_val_for_zin_calc,
        "g_index": torch.tensor([], dtype=torch.long), 
        "r_index": torch.tensor([], dtype=torch.long),
        "l_index": torch.tensor([], dtype=torch.long), 
        "xc_index": torch.tensor([], dtype=torch.long),
        "xl_index": torch.tensor([], dtype=torch.long), 
        "v_index": torch.tensor([], dtype=torch.long),
        "g_value": torch.tensor([], dtype=torch.float64), 
        "r_value": torch.tensor([], dtype=torch.float64),
        "l_value": torch.tensor([], dtype=torch.float64), 
        "xc_value": torch.tensor([], dtype=torch.float64),
        "xl_value": torch.tensor([], dtype=torch.float64),
        "sims": sims
    }
    # --- Base Setup Complete ---


    # --- Iterative Search for Optimal Nd ---
    print(f"\nStarting iterative search for optimal Nd for {CASE_NAME}...")
    optimal_nd_result = None
    max_nd_to_test = num_total_candidates # 尝试所有可能的数量

    start_time_overall = time.time()
    
    # ------------------   暴力搜start    ------------------
    all_nd_results = [] # Store results for all Nd to pick best if no N_no_ok=0 is found
    # 外部循环，从 Nd=0 开始尝试
    for current_nd_test in range(0, max_nd_to_test + 1):
        print(f"\n===== Running GA with NUM_DECAPS_TO_SELECT = {current_nd_test} =====")
        
        best_individual_found, n_no_ok_found = run_ga_for_nd(
            num_decaps_to_select_param=current_nd_test,
            case_name_param=CASE_NAME,
            ga_params=ga_run_params,
            sim_data_dict_base=sim_data_dict_base, # 传递准备好的数据字典
            num_total_candidates=num_total_candidates,
            plot_final_for_this_nd=True # 如果想看每次的结果图，设为 True
        )
        
        print(f"Result for Nd={current_nd_test}: Best N_no_ok={n_no_ok_found}")
        # 存下放置不同数量dtc GA优化后最好的结果
        all_nd_results.append({"Nd": current_nd_test, "n_no_ok": n_no_ok_found, "dtc_indices": best_individual_found})

        # 检查是否找到了可行解 (N_no_ok == 0)
        if n_no_ok_found == 0:
            print(f"Found feasible solution (N_no_ok = 0) with Nd = {current_nd_test}. Stopping search.")
            optimal_nd_result = {
                "Nd": current_nd_test,
                "n_no_ok": n_no_ok_found,
                "dtc_indices": best_individual_found # 保存找到的最佳个体
            }
            break # 找到了就停止外部循环
        

    # ------------------   暴力搜end    ------------------


    # ------------------   二分搜start    ------------------

    # overall_best_result = None # To store the absolute best result if no N_no_ok=0 is found
    # solution_found_nd = -1
    # solution_individual = None
    # solution_n_no_ok = float('inf')

    # low_nd = 0
    # high_nd = num_total_candidates
    
    # # Store all results from binary search path for final selection if no perfect solution
    # # Or to see the trend. Key: Nd, Value: {'n_no_ok': ..., 'dtc_indices': ...}
    # binary_search_results_map = {} 

    # start_time_overall = time.time()
    
    # while low_nd <= high_nd:
    #     mid_nd = low_nd + (high_nd - low_nd) // 2
    #     print(f"\n===== Testing GA with NUM_DECAPS_TO_SELECT = {mid_nd} (Range: [{low_nd}, {high_nd}]) =====")
        
    #     best_individual_found, n_no_ok_found = run_ga_for_nd(
    #         num_decaps_to_select_param=mid_nd,
    #         case_name_param=CASE_NAME, ga_params=ga_run_params,
    #         sim_data_dict_base=sim_data_dict_base, 
    #         num_total_candidates=num_total_candidates,
    #         plot_final_for_this_nd=True, # Plot each GA run in binary search
    #     )
        
        
    #     # Store result of this GA run
    #     binary_search_results_map[mid_nd] = {"n_no_ok": n_no_ok_found, "dtc_indices": best_individual_found}
    #     print(f"Result for Nd={mid_nd}: Best N_no_ok={n_no_ok_found if best_individual_found is not None else 'GA_Run_Failed'}")
        
    #     # if best_individual_found is None and mid_nd > 0 : # GA run failed for this Nd (e.g. Nd too large for candidates)
    #     #     # If GA fails for mid_nd (e.g., returns None for individual), it's like a very bad n_no_ok.
    #     #     # We should probably try smaller Nd, so treat it like n_no_ok > 0.
    #     #     # However, if it's because mid_nd is invalid (e.g. > num_total_candidates),
    #     #     # run_ga_for_nd should handle that and return float('inf') for n_no_ok.
    #     #     # Assuming run_ga_for_nd returns float('inf') for n_no_ok on failure/invalid input.
    #     #      pass # Logic below will handle n_no_ok_found == float('inf')

    #     if n_no_ok_found == 0:
    #         print(f"Feasible solution (N_no_ok = 0) found with Nd = {mid_nd}. Recording and trying smaller Nd.")
    #         solution_found_nd = mid_nd
    #         solution_individual = best_individual_found
    #         solution_n_no_ok = 0
    #         high_nd = mid_nd - 1 # Try to find an even smaller Nd
    #     else: # n_no_ok_found > 0 or GA failed (n_no_ok is inf)
    #         print(f"No feasible solution (N_no_ok = 0) with Nd = {mid_nd}. Need more/different DTCs.")
    #         low_nd = mid_nd + 1 # Need more DTCs

    # total_time_overall = time.time() - start_time_overall
    # print(f"\nBinary search finished in {total_time_overall:.2f} seconds.")

    # if solution_found_nd != -1:
    #     optimal_nd_result = {
    #         "Nd": solution_found_nd,
    #         "n_no_ok": solution_n_no_ok, # Should be 0
    #         "dtc_indices": solution_individual
    #     }
    #     print(f"Optimal Nd found: {solution_found_nd} with N_no_ok = 0.")
    # else: # No N_no_ok=0 found, pick the best from all tested Nds in binary search
    #     print(f"No solution with N_no_ok = 0 found. Selecting best N_no_ok from tested Nd values in binary search.")
    #     if binary_search_results_map:
    #         # Sort by n_no_ok (ascending), then by Nd (ascending)
    #         # Filter out entries where dtc_indices might be None (GA run failed)
    #         valid_results_for_fallback = [
    #             {"Nd": nd, **res} for nd, res in binary_search_results_map.items() if res["dtc_indices"] is not None
    #         ]
    #         if valid_results_for_fallback:
    #             valid_results_for_fallback.sort(key=lambda x: (x["n_no_ok"], x["Nd"]))
    #             optimal_nd_result = valid_results_for_fallback[0]
    #             print(f"Selected best non-ideal solution: Nd={optimal_nd_result['Nd']} with N_no_ok={optimal_nd_result['n_no_ok']}")
    #         else:
    #             optimal_nd_result = None # No valid GA runs completed
    #             print("No valid GA runs completed during binary search.")
    #     else:
    #         optimal_nd_result = None
    #         print("No results from binary search to select from.")
    # ------------------   二分搜end    ------------------
    
    total_time_overall = time.time() - start_time_overall
    print(f"\nIterative search finished in {total_time_overall:.2f} seconds.")

    # --- 处理并输出最终结果 ---
    if optimal_nd_result:
        final_nd = optimal_nd_result["Nd"]
        final_n_no_ok = optimal_nd_result["n_no_ok"]
        final_individual = optimal_nd_result["dtc_indices"]
        print(f"Optimal number of DTCs found: {final_nd}")
        
        # --- 输出 YAML ---
        file_output_ga_dtc = f"data/2025_11/{CASE_NAME}_result_dtc_ga_OptimalNd{final_nd}.yaml"
        result_yaml_output = {
            "dtcs": [], 
            "tsvs": [], 
            # "ga_params": {**ga_run_params, "optimal_Nd": final_nd, "best_n_no_ok": 0, "total_search_time_sec": total_time_overall}
            }
        result_dtc_ga_names = [candidate_dtc_names[i] for i in final_individual] if final_individual else []
        result_yaml_output["tsvs"] = result_tsv_data["tsvs"]
        pattern = R"dtc_(\d+)_(\d+)"
        for dtc_name in result_dtc_ga_names:
             match_result = re.match(pattern, dtc_name)
             if match_result:
                 x = int(match_result.group(1)); y = int(match_result.group(2))
                 result_yaml_output["dtcs"].append((x, y))
        with open(file_output_ga_dtc, "w") as f: yaml.dump(result_yaml_output, f, indent=2)
        print(f"Final optimal GA results saved to {file_output_ga_dtc}")

        # --- 最终绘图 ---
        final_plot_filename = f"{CASE_NAME}_ga_zin_result_OptimalNd{final_nd}.png"
        plot_zin_curves(final_individual, sim_data_dict_base, 
                            f"Final Result (Nd={final_nd}, N_no_ok={final_n_no_ok})", 
                            CASE_NAME, final_nd, final_plot_filename)
        print(f"Plotting logic for final result needs to be called here using plot_zin_curve function.")
        print(f"Final optimal Zin plot would be saved to {final_plot_filename}")

    else:
        print(f"No solution with N_no_ok = 0 found within the tested range up to Nd = {max_nd_to_test}.")
        # (可以选择报告在所有尝试的 Nd 中 N_no_ok 最小的结果)


if __name__ == "__main__":
    # Set random seeds for reproducibility (optional, but good practice for GA)
    # 设置随机数种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42) # Though torch is not directly used for GA randomness here

    # 运行迭代搜索函数
    find_optimal_nd_iteratively()