import pickle
import numpy as np

model_ids = [
    'EnvSimple2D-RobotPointMass',
    'EnvNarrowPassageDense2D-RobotPointMass',
    'EnvDense2D-RobotPointMass',
    'EnvSpheres3D-RobotPanda',
]

# sample_vers = ['mean_ddpm', 'noise_ddpm']
sample_vers = ['noise_ddpm']

# step_sizes = ['3', '2', '1', '0.5', '0.1', '0.05', '0.01', '0.005']
step_sizes = ['7', '5', '3', '2', '1', '0.5', '0.1', '0.05', '0.01', '0.005']

seeds = list(range(0, 100))

# Global variables for pkl files and their labels
# STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_origin.pickle',
#                   '/workspace/scripts/inference/stat_results_latent.pickle']

STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_STOMP.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth1_3.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth2_3.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth3_1_3.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth3_2_3.pickle']

# STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_latent.pickle',
#                   '/workspace/scripts/inference/stat_results_recon.pickle',
#                   '/workspace/scripts/inference/stat_results_recon_back.pickle',
#                   '/workspace/scripts/inference/stat_results_STOMP.pickle']

# STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3_5.pickle',
                #   '/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3_10.pickle',
                #   '/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3_15.pickle']

# STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3Dense_5.pickle',
#                   '/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3Dense_10.pickle',
#                   '/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3Dense_15.pickle']


LABEL_FILES = ['STOMP', 'STOMPsmooth', 'STMOP GP', 'STMOPlinear', 'STOMPcosine']
# LABEL_FILES = ['latent', 'x0', 'ITO', 'STOMP']
# LABEL_FILES = ['trajopt', 'latent']
  
# Function to plot metrics and save figures as .png files
def summerize_metrics(stat_results):
    for model_id in model_ids:
        for i, stat_result in enumerate(stat_results):
            if model_id in stat_result:
                for sample_ver, weights_data in stat_result[model_id].items():
                    if sample_ver in sample_vers:
                        # Ensure 5 is at the beginning if present
                        weights = list(weights_data.keys())
                        success_rate = [weights_data[k]['success_rate'] for k in weights if k in weights_data]
                        
                        max_success_rate_idx = np.argmax(np.array(success_rate))
                        max_weight = weights[max_success_rate_idx]

                        # if i > 0: # For noise blending comparison
                        #     max_weight = '5'
                        print(model_id, sample_ver, LABEL_FILES[i], max_weight)
                        print(stat_result[model_id][sample_ver][max_weight])
                        print()

            
if __name__ == "__main__":
    # Load the two pkl files
    results = []
    for pkl_file in STAT_PKL_FILES:
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)
            results.append(result)

    summerize_metrics(results)