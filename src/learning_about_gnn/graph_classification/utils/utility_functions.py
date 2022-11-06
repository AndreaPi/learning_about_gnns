from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def train_valid_test_loaders(
        dataset: Data,
        p_train: float,
        p_valid: float,
        batch_size: int,
        shuffle=True
) -> Tuple[Data, Data, Data]:

    test_size = (1 - p_train - p_valid) / (1 - p_train)
    train_idx, val_test_idx = train_test_split(range(len(dataset)), train_size=p_train)
    valid_idx, test_idx = train_test_split(range(len(dataset[val_test_idx])), test_size=test_size)

    loader_train = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=shuffle)
    loader_valid = DataLoader(dataset[valid_idx], batch_size=batch_size, shuffle=shuffle)
    loader_test = DataLoader(dataset[test_idx], batch_size=len(test_idx), shuffle=False)

    return loader_train, loader_valid, loader_test


def plot_training_curves(metric, history, plot_dpi, results_dir):
    fig, ax = plt.subplots(dpi=plot_dpi)

    ax.plot(history[f'train_{metric}'], c='steelblue', label='Training')
    ax.plot(history[f'valid_{metric}'], c='orangered', label='Validation')
    ax.grid()
    ax.legend()
    ax.set_title(f'{metric} evolution')
    plt.tight_layout()
    output_filename = Path(results_dir) / f'{metric}_curves.png'
    plt.savefig(output_filename,
                dpi=plot_dpi,  # use 320 dpi for higher-quality images
                format='png',
                transparent=False,
                bbox_inches='tight',
                pad_inches=0)


def compute_batch_confusion_matrix(y_true, y_pred):
    if y_true.is_cuda:
        y_true = y_true.cpu().detach().numpy()
    if y_pred.is_cuda:
        y_pred = y_pred.cpu().detach().numpy()

    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(cm, plot_dpi, labels, results_dir):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    output_filename = Path(results_dir) / 'confusion_matrix.png'
    disp.plot()
    disp.figure_.savefig(output_filename,
                         dpi=plot_dpi,  # use 320 dpi for higher-quality images
                         format='png',
                         transparent=False,
                         bbox_inches='tight',
                         pad_inches=0)
