import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

from defined_networks import given_net_1, Train_History

import os

if not os.path.exists('./results'):
    os.makedirs('./results')

if not os.path.exists('./data'):
    os.makedirs('./data')

# general params
n_epochs = 2
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


def plot_all_figures(hist_case, name):

    if not os.path.exists('./results/'+name):
        os.makedirs('./results/'+name)


    fig = plt.figure()
    plt.plot(hist_case.train_counter, hist_case.train_losses, color='blue')
    plt.scatter(hist_case.test_counter, hist_case.test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples')
    plt.ylabel('Cross Entropy loss value')
    plt.grid()
    fig.savefig('./results/'+name+'/loss_value.png')

    fig2 = plt.figure()
    plt.plot(hist_case.test_counter, hist_case.accuracy_hist_train, color='blue')
    plt.plot(hist_case.test_counter, hist_case.accuracy_hist_test, color='red')
    plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc='lower right')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy [%]')
    plt.grid()
    fig2.savefig('./results/'+name+'/accuracy.png')

    hist_case.train_counter_epoch = [x / len(train_loader.dataset) for x in hist_case.train_counter]

    fig3 = plt.figure()
    plt.plot(hist_case.train_counter_epoch, hist_case.train_losses, color='blue')
    plt.scatter(hist_case.test_counter_epoch, hist_case.test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Epoch number')
    plt.ylabel('Cross Entropy loss value')
    plt.grid()
    fig3.savefig('./results/'+name+'/loss_value_epoch.png')

    fig4 = plt.figure()
    plt.plot(hist_case.test_counter_epoch, hist_case.accuracy_hist_train, color='blue')
    plt.plot(hist_case.test_counter_epoch, hist_case.accuracy_hist_test, color='red')
    plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc='lower right')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy [%]')
    plt.grid()
    fig4.savefig('./results/'+name+'/accuracy_epoch.png')




def train(epoch, net, train_losses , train_counter, name):
  net.train()

  if not os.path.exists('./results/' + name):
      os.makedirs('./results/' + name)

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = net(data)
    loss = F.cross_entropy(output, target)

    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(net.state_dict(), './results/'+name+'/model.pth')
      torch.save(optimizer.state_dict(), './results/'+name+'/optimizer.pth')

def test_validation(net, test_losses, accuracy_hist_test):

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

def test_training(net, train_validation_losses, accuracy_hist_train ):

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
  train_validation_losses.append(train_loss)
  accuracy_hist_train.append(correct.item() * (100 / len(train_loader.dataset)))
  print('\nTraining set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))







### CASE 1 - LEARNING_RATE = 0.01, ERROR = CE

name_case = 'lr_0_01_net_1_CE_'
net = given_net_1()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0)

# initialize the history
hist_case_1 = Train_History(n_epochs = n_epochs, train_dataset_length = len(train_loader.dataset))

# initial validation
test_validation(net=net, test_losses=hist_case_1.test_losses, accuracy_hist_test=hist_case_1.accuracy_hist_test)
test_training(net=net, train_validation_losses=hist_case_1.train_validation_losses,
              accuracy_hist_train=hist_case_1.accuracy_hist_train)

# train and validate
for epoch in range(1, n_epochs + 1):
  train(epoch=epoch, net=net, train_losses=hist_case_1.train_losses,
        train_counter=hist_case_1.train_counter, name=name_case);
  test_validation(net=net, test_losses=hist_case_1.test_losses, accuracy_hist_test=hist_case_1.accuracy_hist_test)
  test_training(net=net, train_validation_losses=hist_case_1.train_validation_losses,
                accuracy_hist_train=hist_case_1.accuracy_hist_train)

# print the final accuracy
print('Classification accuracy on the test set:',
      hist_case_1.accuracy_hist_test[len(hist_case_1.accuracy_hist_test) - 1])

plot_all_figures(hist_case_1, name=name_case)
