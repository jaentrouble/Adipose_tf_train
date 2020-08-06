def lr_no_update(epoch, lr):
    return 0.0

def lr_step(epoch, lr):
    lr = 0.2
    if epoch > 10 :
        lr = 0.02
    if epoch > 20 :
        lr = 0.01
    if epoch > 50 :
        lr = 0.005
    return lr

def lr_step2(epoch, lr):
    if epoch <= 10 :
        lr = (epoch +1) * 0.01
    if epoch > 10 :
        lr = 0.02
    if epoch > 20 :
        lr = 0.01
    if epoch > 50 :
        lr = 0.005
    return lr

def lr_mul_inv(epoch, lr):
    return 0.01 / (epoch+1)

def low_lr(epoch, lr) :
    return 1e-5