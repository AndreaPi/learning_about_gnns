from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


class AddElementSymbols(object):
    """
    Add the symbol of the elements, from the atomic number, as a key of the graph
    """
    MUT_LABEL_ENC = pd.Series(data=["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"])

    def __call__(self, graph: Data):
        graph.symbols = AddElementSymbols.MUT_LABEL_ENC.loc[graph.x.argmax(dim=-1)].values
        return graph


def inspect_mutagenicity_dataset(dataset: TUDataset):
    print(f'Dataset: {dataset}')
    print('====================')
    print(f'number of molecules: {len(dataset)}')
    print(f'number of features: {dataset.num_features}')
    print(f'number of classes: {dataset.num_classes}')
    samples_for_class = [np.sum([1 for graph in dataset if graph.y == i]) for i in range(dataset.num_classes)]
    print(f'samples for class: {samples_for_class}')
    atoms_list = np.concatenate(
        [graph.symbols for graph in dataset]
    )
    print(f'{len(np.unique(atoms_list))} unique atoms: {np.unique(atoms_list)}')


def inspect_molecule(molecule: Data) -> None:
    print('====================')
    print(f'number of nodes: {molecule.num_nodes}')
    print(f'number of edges: {molecule.num_edges}')
    print(f'average node degree: {molecule.num_edges / molecule.num_nodes:.2f}')
    print(f'contains isolated nodes: {molecule.has_isolated_nodes()}')
    print(f'contains self-loops: {molecule.has_self_loops()}')
    print(f'is undirected: {molecule.is_undirected()}')
    print(f'is mutagenic: {"Y" if molecule.y.item() == 0 else "N"}')


def to_molecule(torch_graph: Data) -> nx.Graph:
    """Convert a Pytorch Geometric Data, with attribute _symbols_, into a networkx graph representing a molecule.
    Args:
        torch_graph : Data
            Input Pytorch graph
    Returns:
        nx.Graph:
            Converted graph
    """
    G = to_networkx(
        torch_graph,
        to_undirected=True,
        node_attrs=["symbols"]
    )
    return G


def plot_nx_mol(
    G: nx.Graph,
    results_dir: str,
    edge_mask=None,
    edge_type=None,
    threshold=None,
    drop_isolates=False,
    ax=None,
    plot_basename: str = "molecule.png"
):
    """Draw molecule.
    Args:
        G : nx.Graph
            Graph with _symbols_ node attribute.
        results_dir: str
            folder with results.
        edge_mask : dict, optional
            Dictionary of edge/float items, by default None.
            If given the edges will be color coded. If `threshold` is given,
            `edge_mask` is used to filter edges with mask lower than value.
        edge_type : array of float, optional
            Type of bond encoded as a number, by default None.
            If given, bond width will represent the type of bond.
        threshold : float, optional
            Minimum value of `edge_mask` to include, by default None.
            Only used if `edge_mask` is given.
        drop_isolates : bool, optional
            Whether to remove isolated nodes, by default True if `threshold` is given else False.
        ax : matplotlib.axes.Axes, optional
            Axis on which to draw the molecule, by default None
        plot_basename: str, optional
            the basename of the plot to be written. Full filename = results_dir + plot_basename
    """
    if drop_isolates is None:
        drop_isolates = True if threshold else False
    if ax is None:
        fig, ax = plt.subplots(dpi=120)

    pos = nx.planar_layout(G)
    pos = nx.kamada_kawai_layout(G, pos=pos)

    if edge_type is None:
        widths = None
    else:
        widths = edge_type + 1

    edgelist = G.edges()

    if edge_mask is None:
        edge_color = 'black'
    else:
        if threshold is not None:
            edgelist = [
                (u, v) for u, v in G.edges() if edge_mask[(u, v)] > threshold
            ]

        edge_color = [edge_mask[(u, v)] for u, v in edgelist]

    nodelist = G.nodes()
    if drop_isolates:
        if not edgelist:  # Prevent errors
            print("No nodes left to show !")
            return

        nodelist = list(set.union(*map(set, edgelist)))

    node_labels = {
        node: data["symbols"] for node, data in G.nodes(data=True)
        if node in nodelist
    }

    nx.draw_networkx(
        G, pos=pos,
        nodelist=nodelist,
        node_size=200,
        labels=node_labels,
        width=widths,
        edgelist=edgelist,
        edge_color=edge_color, edge_cmap=plt.cm.Blues,
        edge_vmin=0., edge_vmax=1.,
        node_color='azure',
        ax=ax
    )

    plt.tight_layout()
    full_filename = Path(results_dir) / plot_basename
    plt.savefig(full_filename,
                dpi=120,  # use 320 dpi for higher-quality images
                format='png',
                transparent=False,
                bbox_inches='tight',
                pad_inches=0)


def mask_to_dict(edge_mask, data):
    """
    Convert an `edge_mask` in pytorch geometric format to a networkx compatible
    dictionary (`{(n1, n2) : mask_value}`).
    Multiple edge appearances are averaged.
    """
    edge_mask_dict = defaultdict(float)
    counts = defaultdict(int)

    for val, u, v in zip(edge_mask.to("cpu").numpy(), *data.edge_index):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
        counts[(u, v)] += 1

    for edge, count in counts.items():
        edge_mask_dict[edge] /= count

    return edge_mask_dict


def plot_mol(data, edge_mask=None, **kwargs):
    """Draw a molecule (pytorch geom. Data) with an edge mask from GNN Explainer.
    Wrapper of `plot_nx_mol`.
    Args:
        data : Data
            Molecule of interest
        edge_mask : torch.Tensor, optional
            Edge mask computed by GNNExplainer, by default None
    """
    mol = to_molecule(data)

    if edge_mask is not None:
        edge_mask = mask_to_dict(
            edge_mask,
            data
        )

    plot_nx_mol(mol, edge_mask=edge_mask, **kwargs)
