import pickle
import matplotlib.pyplot as plt

model_ids = [
    'EnvSimple2D-RobotPointMass',
    'EnvNarrowPassageDense2D-RobotPointMass',
    'EnvDense2D-RobotPointMass',
    'EnvSpheres3D-RobotPanda',
]

sample_vers = ['noise_ddpm'] # ['mean_ddpm', 'noise_ddpm']

step_sizes = [3, 1, 0.5, 0.1, 0.05, 0.01, 0.005]

seeds = list(range(0, 300))

STAT_PKL_FILE = '/workspace/scripts/inference/stat_results_recon.pickle'

# Function to plot metrics and save figures as .png files
def plot_metrics(stat_results, metrics):
    
    for model_id, sample_data in stat_results.items():
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), constrained_layout=True)
        fig.suptitle(f'{model_id}', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for sample_ver, weights_data in sample_data.items():
                x_vals = list(weights_data.keys())  # List of weight_grad_cost_collision values
                y_vals = [weights_data[w][metric] for w in x_vals]  # Metric values for this sample_ver

                ax.plot(x_vals, y_vals, label=f'{sample_ver}')
            
            ax.set_title(metric)
            ax.set_xlabel('Step Size')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
        
        # Save the figure with the model_id as the filename
        plt.savefig(f'./{model_id}.png', format='png')
        plt.close(fig)  # Close the figure to avoid memory issues

if __name__ == "__main__":
    with open(STAT_PKL_FILE, 'rb') as f:
        result = pickle.load(f)
    print(result)

    metrics = ['success_rate', 'mean_smoothness', 'mean_path_length', 'averaged_variance', 'mean_intensity']
    plot_metrics(result, metrics)

