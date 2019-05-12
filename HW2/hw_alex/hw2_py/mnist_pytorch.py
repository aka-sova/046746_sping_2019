import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import torch.optim as optim

from defined_networks import given_net_1, found_net_1

import os

if not os.path.exists('./results'):
    os.makedirs('./results')

if not os.path.exists('./data'):
    os.makedirs('./data')

# general params
n_epochs = 8
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10
momentum = 0.5


# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])])

# define dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = 50000
test_size = 10000


train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# plot the images

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig

network_used = 1
# 1 - given_net_1 , 2 - given_net_2, 3 - internet_net

if network_used == 1:
    net = given_net_1()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
elif network_used == 2:
    net = given_net_2()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
else:
    net = found_net_1()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# input = torch.randn(1, 1, 28, 28)
# out = net(input)
# print(out)

train_losses = []
train_counter = []
test_losses = []
train_valiation_losses = []
accuracy_hist_test = []
accuracy_hist_train = []


test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
test_counter_epoch = [i for i in range(n_epochs + 1)]

def train(epoch):
  net.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = net(data)
    if network_used == 1:
        loss = F.cross_entropy(output, target)
    elif network_used == 2:
        loss = F.cross_entropy(output, target)
    else:
        loss = F.nll_loss(output, target)

    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(net.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test_validation():

  net.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = net(data)
      test_loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  accuracy_hist_test.append(correct.item() * (100 / len(test_loader.dataset)))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))



def test_training():

  net.eval()
  train_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in train_loader:
      output = net(data)
      train_loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  train_loss /= len(train_loader.dataset)
  train_valiation_losses.append(train_loss)
  accuracy_hist_train.append(correct.item() * (100 / len(train_loader.dataset)))
  print('\nTraining set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))


test_validation()
test_training()

for epoch in range(1, n_epochs + 1):
  train(epoch)
  test_validation()
  test_training()



fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Number of training examples')
plt.ylabel('Cross Entropy loss value')
plt.grid()
fig.savefig('./results/loss_value.png')


fig2 = plt.figure()
plt.plot(test_counter, accuracy_hist_train, color='blue')
plt.plot(test_counter, accuracy_hist_test, color='red')
plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc='lower right')
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy [%]')
plt.grid()
fig2.savefig('./results/accuracy.png')

print('Classification accuracy on the test set:')
print(*accuracy_hist_test, sep = "\n")


train_counter_epoch = [x / len(train_loader.dataset) for x in train_counter]
# test_counter_epoch

fig3 = plt.figure()
plt.plot(train_counter_epoch, train_losses, color='blue')
plt.scatter(test_counter_epoch, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Epoch number')
plt.ylabel('Cross Entropy loss value')
plt.grid()
fig3.savefig('./results/loss_value_epoch.png')

fig4 = plt.figure()
plt.plot(test_counter_epoch, accuracy_hist_train, color='blue')
plt.plot(test_counter_epoch, accuracy_hist_test, color='red')
plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc='lower right')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy [%]')
plt.grid()
fig4.savefig('./results/accuracy_epoch.png')
