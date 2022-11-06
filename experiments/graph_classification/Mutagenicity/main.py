import torch
from pathlib import Path
from torch_geometric.datasets import TUDataset
import numpy as np
from src.learning_about_gnn.graph_classification.datasets.Mutagenicity import AddElementSymbols, \
    inspect_mutagenicity_dataset, inspect_molecule, plot_mol
from src.learning_about_gnn.graph_classification.utils.utility_functions import train_valid_test_loaders, \
    plot_training_curves, compute_batch_confusion_matrix, plot_confusion_matrix
from src.learning_about_gnn.graph_classification.models.gnn import GNN
from torch.nn.functional import nll_loss
from torch_geometric.nn.models import GNNExplainer
torch.manual_seed(0)
rng = np.random.default_rng(123)

# Parameters
results_dir = Path("./results")
results_dir.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")
p_train = 0.7
p_valid = 0.2
batch_size = 32
hidden_dim = 64
num_epochs = 200
plot_dpi = 120

# Load dataset
dataset = TUDataset(
    root=".",
    name="Mutagenicity",
    transform=AddElementSymbols()
).shuffle()

# Print some info about the dataset
inspect_mutagenicity_dataset(dataset)
print()
molecule_index = rng.choice(len(dataset))
molecule = dataset[molecule_index]
print(f'molecule nr.:{molecule_index}')
inspect_molecule(molecule)

# Draw a molecule using networkx
plot_basename = f'molecule_nr_{molecule_index}.png'
plot_mol(molecule, edge_mask=None, results_dir=results_dir, plot_basename=plot_basename)

# create dataloaders
loader_train, loader_valid, loader_test = train_valid_test_loaders(dataset, p_train, p_valid, batch_size)


# Initialize model
model = GNN(
    num_classes=dataset.num_classes,
    hidden_dim=hidden_dim,
    node_features_dim=dataset.num_node_features,
).to(device)

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters())

# Loss function
loss_function = nll_loss

# Train model
print("\nStart training...")
history = model.fit(loader_train, loader_valid, optimizer=optimizer, loss_function=loss_function, num_epochs=num_epochs)
print("Training complete.")

# Evaluation mode
model.eval()

# Plot training and validation curves
plot_training_curves('loss', history, plot_dpi, results_dir)
plot_training_curves('accuracy', history, plot_dpi, results_dir)

# Compute and plot confusion matrix
batch = next(iter(loader_test))
batch.to(device)
y_pred_test = model.predict_labels(batch.x, batch.edge_index, batch.batch)
y_true_test = batch.y

cm = compute_batch_confusion_matrix(y_true_test, y_pred_test)
labels = ['mutagen', 'nonmutagen']
plot_confusion_matrix(cm, plot_dpi, labels, results_dir)
