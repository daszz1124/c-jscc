import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，确保图表中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def process_visdial_experiment_data():
    """Process VisDial experiment data, calculate CBR, and generate visualization results"""
    # 1. Create experiment data DataFrame
    data = {
        'base_dim': [1536] * 12,
        'latent_dim': [1024, 768, 512, 384, 256, 192, 128, 96, 64, 32, 16, 8],
        'cb_ratio': [None] * 12,  # Initialize as empty, calculate later
        'best_acc': [0.755, 0.763, 0.763, 0.763, 0.763, 0.763, 0.763, 0.763, None, None, None, None],
        'baseline_acc': [0.763] * 8 + [None] * 4
    }
    
    df = pd.DataFrame(data)
    
    df['cb_ratio'] = df['latent_dim'] / df['base_dim']
    
    # 3. Display full data
    print("VisDial experiment full data:")
    print(df.to_string(index=False))
    
    # 4. Data visualization
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy comparison
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x='latent_dim', y='best_acc', marker='o', label='Best Accuracy')
    sns.lineplot(data=df, x='latent_dim', y='baseline_acc', marker='s', label='Baseline Accuracy')
    plt.title('Accuracy Comparison under Different Latent Dimensions')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot CBR vs best accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x='cb_ratio', y='best_acc', marker='o', color='green')
    plt.title('CBR vs Best Accuracy')
    plt.xlabel('Compression Ratio (CBR = latent_dim / base_dim)')
    plt.ylabel('Best Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('visdial_experiment_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'visdial_experiment_results.png'")
    plt.show()
    
    # 5. Calculate and display key statistics
    if df['best_acc'].notna().any():
        print("\nAccuracy Statistics:")
        print(f"Max Accuracy: {df['best_acc'].max():.3f}")
        print(f"Min Accuracy: {df['best_acc'].min():.3f}")
        print(f"Mean Accuracy: {df['best_acc'].mean():.3f}")
    
    return df

if __name__ == "__main__":
    result_df = process_visdial_experiment_data()
    best_acc = result_df['best_acc'].max()
    best_cbr = result_df.loc[result_df['best_acc'] == best_acc, 'CBR'].values
    if len(best_cbr) > 0:
        print(f"\n达到最高准确率 {best_acc:.3f} 对应的CBR值为: {best_cbr[0]:.4f}")
