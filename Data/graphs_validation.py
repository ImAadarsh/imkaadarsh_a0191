import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create plots directory if not exists
os.makedirs('plots', exist_ok=True)

# Load generated data
with open('simulations.json') as f:
    simulations = json.load(f)
    
with open('validation_metrics.json') as f:
    validation = json.load(f)

training_data = pd.read_csv('training_data.csv')
sim_df = pd.json_normalize(simulations)

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# 1. Cost Distribution Analysis (updated)
def plot_cost_distribution():
    plt.figure(figsize=(10,6))
    plt.hist(sim_df['cost_usd'], bins=20, alpha=0.7)
    plt.axvline(sim_df['cost_usd'].max(), color='r', linestyle='--', label='Max Cost')
    plt.axvline(sim_df['cost_usd'].min(), color='g', linestyle='--', label='Min Cost')
    plt.title('Cost Distribution Analysis')
    plt.xlabel('Cost (USD)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/cost_distribution.png')
    plt.close()

# 2. QoS-Cost Tradeoff Analysis
def plot_qos_tradeoff():
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(sim_df['qos_metrics.avg_response_time_ms'], 
                        sim_df['qos_metrics.slo_violation_percent'],
                        c=sim_df['cost_usd'], cmap='viridis')
    plt.colorbar(scatter, label='Cost (USD)')
    plt.axhline(5, color='r', linestyle='--', label='SLO Violation Threshold')
    plt.title('QoS-Cost Tradeoff Analysis')
    plt.xlabel('Average Response Time (ms)')
    plt.ylabel('SLO Violation (%)')
    plt.legend()
    plt.savefig('plots/qos_tradeoff.png')
    plt.close()

# 3. Provider-specific QoS Compliance
def plot_provider_qos():
    plt.figure(figsize=(10,6))
    colors = np.where(sim_df['deployment_config.provider'] == 'IP6', 'g', 'r')
    plt.scatter(sim_df['qos_metrics.avg_response_time_ms'], 
                sim_df['qos_metrics.slo_violation_percent'],
                c=colors, alpha=0.6)
    plt.axhline(5, color='r', linestyle='--')
    plt.title('Provider-specific QoS Compliance (Green=IP6)')
    plt.xlabel('Response Time (ms)')
    plt.ylabel('SLO Violation (%)')
    plt.savefig('plots/provider_qos.png')
    plt.close()

# 4. ECDF Plots for Response Times
def plot_ecdf():
    plt.figure(figsize=(10,6))
    for provider in sim_df['deployment_config.provider'].unique():
        subset = sim_df[sim_df['deployment_config.provider'] == provider]
        x = np.sort(subset['qos_metrics.avg_response_time_ms'])
        y = np.arange(1, len(x)+1)/len(x)
        plt.step(x, y, label=provider)
    
    plt.title('ECDF of Response Times by Provider')
    plt.xlabel('Response Time (ms)')
    plt.ylabel('ECDF')
    plt.legend()
    plt.savefig('plots/ecdf_response.png')
    plt.close()

# 5. Permutation Importance Plot (Simulated based on paper data)
def plot_feature_importance():
    features = {
        'MRS Cost': ['NW Egress', 'Provider4', 'Provider1', 'Provider3'],
        'Video Cost': ['NW Egress', 'Provider1', 'Provider4', 'Provider3'],
        'MRS QoS': ['Users', 'RAM Used', 'Disk Read', 'NW Egress'],
        'Video QoS': ['Users', 'NW Egress', 'RAM Used', 'Disk Read']
    }
    
    importance = {
        'MRS Cost': [1.03, 0.95, 0.85, 0.69],
        'Video Cost': [1.62, 0.99, 0.88, 0.63],
        'MRS QoS': [2.04, 0.86, 0.50, 0.41],
        'Video QoS': [2.29, 0.03, 0.00, 0.00]
    }
    
    fig, axs = plt.subplots(2, 2, figsize=(18,12))
    for idx, (title, feat) in enumerate(features.items()):
        ax = axs[idx//2, idx%2]
        sns.barplot(x=importance[title], y=feat, ax=ax)
        ax.set_title(f'Feature Importance - {title}')
        ax.set_xlabel('Permutation Importance Weight')
    
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

# 6. Model Accuracy Comparison
def plot_model_accuracy():
    metrics = ['MAE', 'MAPE', 'RMSE', 'R2']
    models = ['Linear', 'Lasso', 'LARS', 'MARS', 'ANN']
    
    # Simulated data based on paper's Table 5
    data = {
        'MRS Response': [243.39, 476.62, 181.11, 181.47, 221.05],
        'MRS Cost': [142.98, 113.68, 99.00, 106.25, 124.91],
        'Video Response': [266.80, 287.26, 168.65, 104.87, 221.05],
        'Video Cost': [331.24, 234.18, 123.59, 72.77, 531.55]
    }
    
    fig, axs = plt.subplots(2, 2, figsize=(18,12))
    for idx, (title, values) in enumerate(data.items()):
        ax = axs[idx//2, idx%2]
        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(f'Model Comparison - {title}')
        ax.set_ylabel(metrics[idx%2] if idx%2 < 2 else metrics[2 + idx%2])
    
    plt.tight_layout()
    plt.savefig('plots/model_accuracy.png')
    plt.close()

# 7. Optimization Gain Visualization
def plot_optimization_gain():
    gains = {
        'MRS Dynamic': 95.39,
        'MRS Growth': 94.63,
        'MRS Burst': 94.58,
        'Video Dynamic': 92.93,
        'Video Growth': 92.99,
        'Video Burst': 92.93
    }
    
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(gains.keys()), y=list(gains.values()))
    plt.title('Cost Reduction vs Max Price by Scenario')
    plt.ylabel('Reduction (%)')
    plt.xticks(rotation=45)
    plt.savefig('plots/optimization_gain.png')
    plt.close()

# Generate all plots
plot_cost_distribution()
plot_qos_tradeoff()
plot_provider_qos()
plot_ecdf()
plot_feature_importance()
plot_model_accuracy()
plot_optimization_gain()

print("Generated 7 plots in 'plots' directory:")
print("- cost_distribution.png")
print("- qos_tradeoff.png")
print("- provider_qos.png")
print("- ecdf_response.png")
print("- feature_importance.png")
print("- model_accuracy.png")
print("- optimization_gain.png")
