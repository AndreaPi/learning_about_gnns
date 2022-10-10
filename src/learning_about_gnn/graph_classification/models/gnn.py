import torch.nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.pool import global_add_pool
from tqdm.auto import tqdm
from typing import Tuple
from torchmetrics.functional import accuracy


class GNN(torch.nn.Module):
    def __init__(
            self,
            hidden_dim,
            node_features_dim,
            edge_features_dim=None
    ):
        super(GNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = GraphConv(node_features_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x

    def __update_loss_and_metric(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
            batch_size: int,
            loss_fun,
            loss_accumulator: float,
            accuracy_accumulator: float
    ) -> Tuple[torch.tensor, float, float]:

        loss = loss_fun(logits, target.float())
        acc = accuracy(logits, target, threshold=0.)  # threshold=0. if logits, 0.5 if probas
        loss_accumulator += loss * batch_size
        accuracy_accumulator += acc * batch_size
        return loss, loss_accumulator, accuracy_accumulator

    def __epoch_loop(
            self,
            loader,
            loss_function,
            losses,
            accuracies,
            n,
            epoch,
            device,
            optimizer=None
    ):
        epoch_loss, epoch_acc = 0, 0

        for batch in tqdm(loader, leave=False):
            batch.to(device)
            batch_size = torch.max(batch.batch) + 1

            out = self.forward(batch.x, batch.edge_index, batch.batch)
            target = batch.y.unsqueeze(1)

            loss, epoch_loss, epoch_acc = self.__update_loss_and_metric(
                out,
                target,
                batch_size,
                loss_function,
                epoch_loss,
                epoch_acc
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            losses[epoch] = epoch_loss/n
            accuracies[epoch] = epoch_acc/n

        return losses, accuracies

    def fit(self, loader_train, loader_valid, optimizer, loss_function, num_epochs, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        train_losses = torch.zeros(num_epochs)
        train_accuracies = torch.zeros(num_epochs)
        valid_losses = torch.zeros(num_epochs)
        valid_accuracies = torch.zeros(num_epochs)
        n_train = len(loader_train.dataset)
        n_valid = len(loader_valid.dataset)

        for epoch in tqdm(range(num_epochs)):

            # loop over training set and update weights
            train_losses, train_acc = self.__epoch_loop(loader_train, loss_function, train_losses,
                                                        train_accuracies, n_train, epoch, device, optimizer)
            # loop over validation set and DON'T update weights
            valid_losses, valid_acc = self.__epoch_loop(loader_valid, loss_function, valid_losses,
                                                        valid_accuracies, n_valid, epoch, device)

            print()
            print(f"Epoch: {epoch+1}/{num_epochs}\n"
                  f"training loss: {train_losses[epoch]}, validation loss: {valid_losses[epoch]}\n"
                  f"training accuracy: {train_accuracies[epoch]}, validation accuracy: {valid_accuracies[epoch]}")

        return {
            "train_loss": train_losses,
            "train_accuracy": train_accuracies,
            "valid_loss": valid_losses,
            "valid_accuracy": valid_accuracies,
        }
