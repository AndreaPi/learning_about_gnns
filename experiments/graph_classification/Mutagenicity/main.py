import torch
from pathlib import Path
from torch_geometric.datasets import TUDataset
import numpy as np
from src.learning_about_gnn.graph_classification.datasets.Mutagenicity import AddElementSymbols, \
    inspect_mutagenicity_dataset, inspect_molecule, plot_mol
from src.learning_about_gnn.graph_classification.utils.utility_functions import train_valid_test_loaders, \
    plot_training_curves, compute_batch_confusion_matrix, plot_confusion_matrix
from src.learning_about_gnn.graph_classification.models.gnn import GNN
from torch.nn.functional import binary_cross_entropy_with_logits


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
molecule_index = np.random.choice(len(dataset))
molecule = dataset[molecule_index]
print(f'molecule nr.:{molecule_index}')
inspect_molecule(molecule)

# Draw a molecule using networkx
plot_mol(molecule, edge_mask=None, results_dir=results_dir, index=molecule_index)

# create dataloaders
loader_train, loader_valid, loader_test = train_valid_test_loaders(dataset, p_train, p_valid, batch_size)


# Initialize model
model = GNN(
    hidden_dim=hidden_dim,
    node_features_dim=dataset.num_node_features,
).to(device)

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters())

# Loss function
loss_function = binary_cross_entropy_with_logits

# Train model
print("\nStart training...")
history = model.fit(loader_train, loader_valid, optimizer=optimizer, loss_function=loss_function, num_epochs=num_epochs)
print("Training complete.")

# Plot training and validation curves
plot_training_curves('loss', history, plot_dpi, results_dir)
plot_training_curves('accuracy', history, plot_dpi, results_dir)

# Compute and plot confusion matrix
cm = compute_batch_confusion_matrix(model, loader_test, device)
labels = ['mutagen', 'nonmutagen']
plot_confusion_matrix(cm, plot_dpi, labels, results_dir)
