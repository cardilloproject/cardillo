def split2D(idx, shape):
    i = idx % shape[0]
    j = idx // shape[0]
    return i, j


def flat2D(i, j, shape):
    return j * shape[0] + i


def split3D(idx, shape):
    i = idx % shape[0]
    j = (idx // shape[0]) % shape[1]
    k = idx // (shape[0] * shape[1])
    return i, j, k


def flat3D(i, j, k, shape):
    return k * (shape[0] * shape[1]) + j * shape[0] + i


if __name__ == "__main__":
    shape = (2, 3)
    idx = 4
    split = split2D(idx, shape)
    print(f"split: {split}")
    flat = flat2D(*split, shape)
    print(f"flat: {flat}")

    assert flat == idx

    shape = (2, 3)
    idx = 4
    split = split3D(idx, shape)
    print(f"split: {split}")
    flat = flat3D(*split, shape)
    print(f"flat: {flat}")

    assert flat == idx
