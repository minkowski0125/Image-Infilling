def get_shift(mask):
    x, y = mask.shape[:2]
    mask_y, mask_x = (mask.sum(axis=0) == 0).tolist(), (mask.sum(axis=1) == 0).tolist()
    x_0 = mask_x.index(False)
    x_1 = x - mask_x[::-1].index(False)
    y_0 = mask_y.index(False)
    y_1 = y - mask_y[::-1].index(False)
    return x_0, x_1, y_0, y_1