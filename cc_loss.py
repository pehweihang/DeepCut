import torch


def loss(A, S, num_clusters):
    """
    loss calculation, relaxed form of Normalized-cut
    @param A: Adjacency matrix of the graph
    @param S: Polled graph (argmax of S)
    @return: loss value
    """
    # cc loss
    X = torch.matmul(S, S.t())
    cc_loss = -torch.sum(A * X)

    return cc_loss
