import os
import pickle
import numpy as np

model_ids = [
    'EnvSimple2D-RobotPointMass',
    'EnvNarrowPassageDense2D-RobotPointMass',
    'EnvDense2D-RobotPointMass',
    'EnvSpheres3D-RobotPanda',
]

# sample_vers = ['mean_ddpm', 'noise_ddpm']

# step_sizes = [25, 20, 15, 10, 5, 3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005]
step_sizes = [7, 5, 3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005]
# step_sizes = [5, 3, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005]

sample_vers = ['noise_ddpm']
# sample_nums = [5, 10, 15, 20, 25, 30, 40, 50]

seeds = list(range(0, 300))

# SAVE_DIR = '/data/MPDresultsVanillaSTOMPFixed3_Dense' # Watch out : Recon, _, 
SAVE_DIR = '/data/MPDresultsVanillaSTOMPFixed3'
PKL_FILE = 'results_data_dict.pickle'
DATA_SAVE_DIR = '/workspace/scripts/inference/'

if __name__ == "__main__":
    # Calculate each stat for each env and sample version
    stat_results = {}
    for model_id in model_ids:
        stat_results[model_id] = {}
        for sample_ver in sample_vers:
            stat_results[model_id][sample_ver] = {}

            for stomp_iter in [15]:
                
                for step_size in step_sizes:
                    str_step_size = f'{step_size:.2g}'
                    success = 0
                    time_elapsed = 0
                    smoothness = 0
                    path_length = 0
                    intensity = 0
                    variance = 0
                    
                    if model_id == 'EnvSpheres3D-RobotPanda' and step_size == 7:
                        continue     

                    for seed in seeds:
                        results_dir = os.path.join(SAVE_DIR, model_id, sample_ver, str(stomp_iter), str(seed), str_step_size)
                        # results_dir = os.path.join(SAVE_DIR, model_id, sample_ver, str(seed), str_step_size)
                        # results_dir = os.path.join(SAVE_DIR, model_id,  str(seed), str(sample_num))
                        # print(results_dir)

                        file_path = os.path.join(results_dir, PKL_FILE)
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                result = pickle.load(f)
                            case_success = result['success_free_trajs']
                            intensity += result['collision_intensity_trajs']
                        else:
                            continue

                        success += case_success
                        
                        if case_success == 1:
                            # print("time: ", result['t_total'])
                            # print("path length: ", result['cost_path_length_trajs_final_free'] )
                            # print("smoothness: ", result['cost_smoothness_trajs_final_free'])
                            # print("intensity: ", result['collision_intensity_trajs'])
                            # print("variance: ", result['variance_waypoint_trajs_final_free'])

                            time_elapsed += result['t_total']
                            path_length += result['cost_path_length_trajs_final_free'] 
                            smoothness += result['cost_smoothness_trajs_final_free']
                            
                            if not np.isnan(result['variance_waypoint_trajs_final_free']):
                                # if none, zero variance
                                variance += result['variance_waypoint_trajs_final_free']
                        
                    stat_results[model_id][sample_ver][str_step_size] =\
                                                        {'success_rate': success / len(seeds) * 100,
                                                        'mean_time_elapsed': time_elapsed / success if success != 0 else None,
                                                        'mean_path_length': path_length / success if success != 0 else None,
                                                        'mean_smoothness': smoothness / success if success != 0 else None,
                                                        'averaged_variance': variance / success if success != 0 else None,
                                                        'mean_intensity': intensity / len(seeds) * 100}
        
        print(stat_results)
        with open(DATA_SAVE_DIR + 'stat_results_test.pickle', 'wb') as f:
            pickle.dump(stat_results, f, protocol=pickle.HIGHEST_PROTOCOL) 
                    
