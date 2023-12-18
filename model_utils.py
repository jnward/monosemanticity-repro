def count_params(m):
    print(sum(p.numel() for p in m.parameters()))