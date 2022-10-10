from torch_geometric.datasets import TUDataset
from src.learning_about_gnn.graph_classification.utils.utility_functions import train_valid_test_loaders


def test_train_valid_test_loaders():
    dataset = TUDataset(
        root=".",
        name="Mutagenicity",
    )
    p_train = 0.7
    p_valid = 0.2
    batch_size = 32

    n = len(dataset)
    n_train = int(p_train * n)
    n_valid = int(p_valid * n)
    n_test = n - n_train - n_valid

    loader_train, loader_valid, loader_test = train_valid_test_loaders(dataset, p_train, p_valid,
                                                                       batch_size, shuffle=False)

    assert len(loader_train.dataset) == n_train
    assert len(loader_valid.dataset) == n_valid
    assert len(loader_test.dataset) == n_test
    assert len(next(iter(loader_train))) == batch_size
