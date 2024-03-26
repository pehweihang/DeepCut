import torch


def loss(A, S, num_clusters):
    """
    loss calculation, relaxed form of Normalized-cut
    @param A: Adjacency matrix of the graph
    @param S: Polled graph (argmax of S)
    @return: loss value
    """
    # cut loss
    A_pool = torch.matmul(torch.matmul(A, S).t(), S)
    num = torch.trace(A_pool)

    D = torch.diag(torch.sum(A, dim=-1))
    D_pooled = torch.matmul(torch.matmul(D, S).t(), S)
    den = torch.trace(D_pooled)
    mincut_loss = -(num / den)

    # orthogonality loss
    St_S = torch.matmul(S.t(), S)
    I_S = torch.eye(num_clusters)
    ortho_loss = torch.norm(St_S / torch.norm(St_S) - I_S / torch.norm(I_S))

    return mincut_loss + ortho_loss
