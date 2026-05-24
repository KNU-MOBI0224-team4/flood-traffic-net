from graph_loader import load_adjacency_matrix

A_hat = load_adjacency_matrix(
    r"C:\Users\graph\Desktop\flood-traffic-net\data\data_train\d7_active_giant_2016_01\graph\adjacency_matrix.csv"
)

print(A_hat.shape)
print(A_hat.dtype)