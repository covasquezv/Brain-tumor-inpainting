import numpy as np
import matplotlib.pylab as plt

loss = []
val_loss = []

paths = ['/home/cota/Documents/TESIS/Keras/codes/code_GRAY/logs/coronal/phase1/', '/home/cota/Documents/TESIS/Keras/codes/code_GRAY/logs/coronal/phase2/']

for path in paths:
    l = np.load(path + 'loss.npy')
    v = np.load(path + 'val_loss.npy')

    for n in range(len(l)):

        loss.append(l[n])
        val_loss.append(v[n])


stage = len(np.load('/home/cota/Documents/TESIS/Keras/codes/code_GRAY/logs/coronal/phase1/loss.npy'))


# loss_1 = np.load('/home/cota/Documents/TESIS/Keras/circ_irr/step1_irr_circ/loss.npy')
# val_loss_1 = np.load('/home/cota/Documents/TESIS/Keras/circ_irr/step1_irr_circ/val_loss.npy')
# loss_2 = np.load('/home/cota/Documents/TESIS/Keras/circ_irr/step2_irr_circ/loss_2.npy')
# val_loss_2 = np.load('/home/cota/Documents/TESIS/Keras/circ_irr/step2_irr_circ/val_loss_2.npy')
#
# stage_ = len(loss_1)
#
# loss_, val_loss_ = [],[]
#
# for i in range(len(loss_1)):
#     loss_.append(loss_1[i])
#     val_loss_.append(val_loss_1[i])
#
# for j in range(len(loss_2)):
#     loss_.append(loss_2[j])
#     val_loss_.append(val_loss_2[j])

plt.figure(1, figsize=(15, 10))
plt.plot(loss)
plt.plot(val_loss)
plt.axvline(x=stage, color = 'r')
plt.legend(('train', 'val'))
plt.ylabel('loss (log)')
plt.xlabel('epoch')
plt.yscale('log')
plt.grid()
# plt.yticks(np.arange(0, 1, step=0.001))
plt.savefig('/home/cota/Documents/TESIS/Keras/codes/code_GRAY/logs/coronal/loss_log.png')
