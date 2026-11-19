"""Neural building blocks of the graph methods.

  graph         : relation graph of the scene (no torch_geometric)
  preprocessing : running standardization and the ego context
  gnn           : attention GNN encoder (needs torch_geometric)
  networks      : policy and value networks built from networks.yaml specs
  mlp           : MLP
  distributions : squashed Gaussian (TanhNormal)
"""
