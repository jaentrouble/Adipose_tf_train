def lr_step(epoch, lr):
    lr = 0.2
    if epoch > 10 :
        lr = 0.02
    if epoch > 20 :
        lr = 0.01
    if epoch > 50 :
        lr = 0.005
    return lr