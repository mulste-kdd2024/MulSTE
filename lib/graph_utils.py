from .utils import log_string
import numpy as np
import torch

def calculate_laplacian_with_self_loop(matrix: torch.Tensor):
    """
    matrix: (num_nodes, num_nodes)
    
    return
    normalized_laplacian: (num_nodes, num_nodes)
    """
    diag = torch.diag(matrix)
    if torch.all(torch.eq(diag, torch.zeros(len(diag)).type_as(matrix))):                               
        # A_hat = A + I 
        matrix = matrix + torch.eye(matrix.size(0)).type_as(matrix)                                              
    # node degree
    row_sum = matrix.sum(1)
    # inverse square root of node degree
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()                                                     
    # if node degree is inf, assign the value to 0.0 
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    # D^(-1/2)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # (A_hat D^(-1/2))^T D^(-1/2) -> D^(-1/2) A_hat D^(-1/2)
    normalized_laplacian = (matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt))       
    
    return normalized_laplacian

def multi_graph_construction(args):
    
    print("-------------------------------------------------------------")
    print("Construct multi-view graph...")
    device = 'cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu' # 'cuda'/'gpu'  or 'cpu'
    # print('device=', device)
    print("Construct distance-aware graph")
    adj_distance = np.load(args.adj_distance_path)
    adj_distance = torch.FloatTensor(adj_distance/np.max(adj_distance)).to(device)
    adj_distance = calculate_laplacian_with_self_loop(adj_distance).unsqueeze(0)

    print("Construct neighbor-aware graph")
    adj_neighbor = np.load(args.adj_neighbor_path)
    adj_neighbor = torch.FloatTensor(adj_neighbor/np.max(adj_neighbor)).to(device)
    adj_neighbor = calculate_laplacian_with_self_loop(adj_neighbor).unsqueeze(0)

    print("Construct type-similarity graph")
    adj_road_sim = np.load(args.adj_road_sim_path)
    adj_road_sim = torch.FloatTensor(adj_road_sim/np.max(adj_road_sim)).to(device)
    adj_road_sim = calculate_laplacian_with_self_loop(adj_road_sim).unsqueeze(0)

    print("Construct crowd-similarity graph")
    adj_crowd_sim = np.load(args.adj_crowd_sim_path)
    adj_crowd_sim = torch.FloatTensor(adj_crowd_sim/np.max(adj_crowd_sim)).to(device)
    adj_crowd_sim = calculate_laplacian_with_self_loop(adj_crowd_sim).unsqueeze(0)

    adj = torch.cat([adj_distance, adj_neighbor, adj_road_sim, adj_crowd_sim], dim = 0)
    
    return adj
    
    