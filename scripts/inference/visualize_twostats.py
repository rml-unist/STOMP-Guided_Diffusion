import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rc('axes', labelsize=15)   # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=11)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=15)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=15)  # 범례 폰트 크기

model_ids = [
    # 'EnvSimple2D-RobotPointMass',
    # 'EnvNarrowPassageDense2D-RobotPointMass',
    'EnvDense2D-RobotPointMass',
    'EnvSpheres3D-RobotPanda',
]

# sample_vers = ['mean_ddpm', 'noise_ddpm']
sample_vers = ['noise_ddpm']

# step_sizes = ['3', '2', '1', '0.5', '0.1', '0.05', '0.01', '0.005']
# step_sizes = ['7', '5', '3', '2', '1', '0.5', '0.1', '0.05', '0.01', '0.005']
step_sizes = ['0.005', '0.01', '0.05', '0.1', '0.5', '1', '2', '3']

seeds = list(range(0, 100))

# Global variables for pkl files and their labels
STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_STOMP.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth1_3.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth2_3.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth3_1_3.pickle',
                  '/workspace/scripts/inference/stat_results_STOMP_smooth3_2_3.pickle']

LABEL_FILES = ['SGD', 'SGDsmooth', 'SGD GP', 'SGDlinear', 'SGDcosine']
COLORS = ['orange', 'forestgreen', 'deeppink', 'steelblue', 'black']

# STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_latent.pickle',
#                   '/workspace/scripts/inference/stat_results_recon.pickle',
#                   '/workspace/scripts/inference/stat_results_recon_back.pickle',
#                   '/workspace/scripts/inference/stat_results_STOMP.pickle']

# LABEL_FILES = ['latent', 'FUG', 'ITO', 'SGD'] 
# COLORS = ['green', 'blue', 'red', 'orange']

# STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3Dense_5.pickle',
#                   '/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3Dense_10.pickle',
#                   '/workspace/scripts/inference/stat_results_VanillaSTOMPFixed3Dense_15.pickle']
# LABEL_FILES = ['5', '10', '15']
# COLORS = ['green', 'blue', 'red']

# STAT_PKL_FILES = ['/workspace/scripts/inference/stat_results_latent.pickle',
#                   '/workspace/scripts/inference/stat_results_origin.pickle']
# LABEL_FILES = ['latent', 'trajopt']
# COLORS = ['green', 'blue']

metrics_limits = {'EnvDense2D-RobotPointMass' : [(75, 90), (0, 15), (1.9, 2.5), (1, 4), (3, 4)],
                  'EnvSpheres3D-RobotPanda' : [(55, 100), (0, 25), (5.5, 7.5), (10, 45), (5, 7)]}

# Function to plot metrics and save figures as .png files

pretty_x = {'0.005':'5e-3', '0.01':'1e-2', '0.05':'5e-2'}
def make_x_pretty(x_vals):
    len_x = len(x_vals)
    for i in range(len_x):
        if x_vals[i] in pretty_x.keys():
            x_vals[i] = pretty_x[x_vals[i]]
    return x_vals

def plot_metrics(stat_results, metrics):
    for model_id in model_ids:
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(4*5, 5), constrained_layout=True)
        # fig.suptitle(f'{model_id}', fontsize=12)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.yaxis.set_major_locator(MaxNLocator(prune=None, steps=[2, 5]))

            for j, stat_result in enumerate(stat_results):
                # Plot data from the pkl file
                if model_id in stat_result:
                    for sample_ver, weights_data in stat_result[model_id].items():
                        if sample_ver in sample_vers:

                            # Ensure 5 is at the beginning if present
                            x_vals = list(weights_data.keys())
                            x_vals.reverse()
                            # x_vals = sorted(list(weights_data.keys()), reverse=True)
                            # Fetch y values corresponding to the sorted x values
                            y_vals = [weights_data[k][metric] for k in x_vals if k in weights_data]
                            print(y_vals)

                            if model_id != 'EnvSpheres3D-RobotPanda':
                                if len(x_vals) < len(step_sizes):
                                    x_vals = x_vals + step_sizes[-(len(step_sizes)-len(x_vals)):]
                                    y_vals = y_vals + [None] * (len(step_sizes)-len(y_vals))
                            else:
                                pass
                                # if '5' not in x_vals:
                                #     x_vals = sorted(x_vals) + ['5']
                                #     y_vals = y_vals + [None]
                                # if '7' in x_vals:
                                #     x_vals = x_vals[:-1]
                                #     y_vals = y_vals[:-1]

                            # if len(x_vals) > len(step_sizes):
                            #     x_vals = x_vals[(len(x_vals) - len(step_sizes)):]
                            #     y_vals = y_vals[(len(y_vals) - len(step_sizes)):]

                            # if '7' not in x_vals:
                            #     x_vals = ['7'] + sorted(x_vals, reverse=True)
                            #     y_vals = [None] + y_vals

                            ls = '-' if sample_ver == 'noise_ddpm' else '--'
                            x_vals = make_x_pretty(x_vals)
                            ax.plot(x_vals, y_vals, label=f'{LABEL_FILES[j]}', c=COLORS[j], ls=ls, linewidth=3)
                            ax.set_ylim(metrics_limits[model_id][i][0], metrics_limits[model_id][i][1])
            
            title = metric
            ax.set_title(title, fontsize=18)
            ax.set_xlabel('step size')
            # ax.set_ylabel(metric.replace('_', ' ').title())
            if i == 0:
                ax.legend()
            
        for j in range(4, len(metrics)-1, -1):
            axes[j].axis('off')
        
        # Save the figure with the model_id as the filename
        plt.savefig(f'./{model_id}_Figure9.eps', format='eps')
        plt.close(fig)  # Close the figure to avoid memory issues

if __name__ == "__main__":
    # Load the two pkl files
    results = []
    for pkl_file in STAT_PKL_FILES:
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)
            results.append(result)

    metrics = ['success_rate', 'mean_intensity', 'mean_smoothness', 'averaged_variance', 'mean_path_length']
    # metrics = ['success_rate', 'mean_intensity', 'mean_smoothness', 'averaged_variance']
    # metrics = ['success_rate', 'mean_intensity']
    plot_metrics(results, metrics)