def make_factor(labels):
    """
    Convert arbitrary labels to integer levels {0, 1, ..., K-1}.
    """
    uniques = {}
    levels = []
    for lab in labels:
        if lab not in uniques:
            uniques[lab] = len(uniques)
        levels.append(uniques[lab])
    return levels, uniques

def one_hot(labels_int, num_classes):
    """
    Convert list of integer labels to simple one-hot rows.
    """
    out = []
    for k in labels_int:
        row = [0.0] * num_classes
        row[k] = 1.0
        out.append(row)
    return out