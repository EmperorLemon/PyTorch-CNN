import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tensorboardX import SummaryWriter
import seaborn as sns

def visualize_results(conf_matrix, mae, writer: SummaryWriter):
    # Print results
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"Shape of confusion matrix: {conf_matrix.shape}")

    print(f"\nMean Absolute Error: {mae:.4f}")
    
    # Log to TensorBoardX
    writer.add_scalar("Test/MAE", mae)

    # Create and log confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", square=True, cbar=True, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    writer.add_figure("Confusion Matrix", fig)
    plt.close(fig)