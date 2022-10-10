import torch
from src.learning_about_gnn.graph_classification.models.gnn import GNN
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F


hidden_dim = 64
num_node_features = 14


def binary_logistic_loss(logits, targets):
    first = -targets.matmul(F.logsigmoid(logits))
    second = -(1 - targets).matmul(F.logsigmoid(logits) - logits)
    return (first + second)/logits.shape[0]


def unnormalized_accuracy(logits, targets):
    y_hat = (logits > 0).int()
    return sum(y_hat == targets)


def test_update_loss_and_metric():

    loss_fun = binary_cross_entropy_with_logits
    loss_accumulator = 0
    accuracy_accumulator = 0

    out = torch.tensor([-1., -2., 0.5, 1.1, 0.001, -1.7, .7, .354, .8, -.12, 1.37, -1.85, 2.17, -.56, .68, .12])
    target = torch.tensor([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0])

    batch_size = out.shape[0]

    GT_loss = binary_logistic_loss(out, target.float())
    GT_unormalized_accuracy = unnormalized_accuracy(out, target)

    model = GNN(
        hidden_dim=hidden_dim,
        node_features_dim=num_node_features,
    )

    loss, epoch_loss, epoch_acc = model._GNN__update_loss_and_metric(
        out,
        target,
        batch_size,
        loss_fun,
        loss_accumulator,
        accuracy_accumulator
    )
    assert torch.isclose(loss, GT_loss)
    assert torch.isclose(epoch_loss, GT_loss * batch_size)
    assert (epoch_acc == GT_unormalized_accuracy)
