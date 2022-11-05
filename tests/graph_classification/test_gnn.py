import torch
from src.learning_about_gnn.graph_classification.models.gnn import GNN
from torch.nn import LogSoftmax
from torch.nn.functional import nll_loss


hidden_dim = 64
num_node_features = 14


def negative_loglikelihood_loss(logprobas, targets):
    num = -logprobas.gather(1, targets.view(-1, 1))
    denom = logprobas.shape[0]
    return torch.sum(num/denom)


def unnormalized_accuracy(out, targets):
    y_hat = torch.argmax(out, dim=1)  # Use the class with the highest probability/log-probability (log is monotone)
    return sum(y_hat == targets)


def test_update_loss_and_metric():

    loss_fun = nll_loss
    loss_accumulator = 0
    accuracy_accumulator = 0

    logits = torch.tensor([-1., -2., 0.5, 1.1, 0.001, -1.7, .7, .354, .8, -.12, 1.37, -1.85, 2.17, -.56, .68, .12])
    logits = torch.column_stack((logits, -logits))
    logsoftmax = LogSoftmax(dim=1)
    logprobas = logsoftmax(logits)

    target = torch.tensor([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0])

    batch_size = logprobas.shape[0]

    GT_loss = negative_loglikelihood_loss(logprobas, target)
    GT_unnormalized_accuracy = unnormalized_accuracy(logprobas, target)

    model = GNN(
        num_classes=2,
        hidden_dim=hidden_dim,
        node_features_dim=num_node_features,
    )

    loss, epoch_loss, epoch_acc = model._GNN__update_loss_and_metric(
        logprobas,
        target,
        batch_size,
        loss_fun,
        loss_accumulator,
        accuracy_accumulator
    )
    assert torch.isclose(loss, GT_loss)
    assert torch.isclose(epoch_loss, GT_loss * batch_size)
    assert (epoch_acc == GT_unnormalized_accuracy)
