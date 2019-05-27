import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

from defined_networks import given_net_1, given_net_2, Train_History

import os

from datetime import datetime

now = datetime.now()

cur_time = now.strftime("_%d_%H_%M/")
results_dirname = './results/'+cur_time
print('Current time: ' + cur_time)


if not os.path.exists(results_dirname):
    os.makedirs(results_dirname)

if not os.path.exists(results_dirname + 'comparison'):
    os.makedirs(results_dirname + 'comparison')



if not os.path.exists('./data'):
    os.makedirs('./data')

# general params
batch_size_train = 128
batch_size_test = 1000
log_interval = 10


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


def plot_all_figures(hist_case, results_dirname, name):

    if not os.path.exists(results_dirname+name):
        os.makedirs(results_dirname+name)


    fig = plt.figure()
    plt.plot(hist_case.train_counter, hist_case.train_losses, color='blue')
    plt.scatter(hist_case.test_counter, hist_case.test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples')
    plt.ylabel('Cross Entropy loss value')
    plt.grid()
    fig.savefig(results_dirname+name+'/loss_value.png')

    fig2 = plt.figure()
    plt.plot(hist_case.test_counter, hist_case.accuracy_hist_train, color='blue')
    plt.plot(hist_case.test_counter, hist_case.accuracy_hist_test, color='red')
    plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc='lower right')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy [%]')
    plt.grid()
    fig2.savefig(results_dirname+name+'/accuracy.png')

    hist_case.train_counter_epoch = [x / len(train_loader.dataset) for x in hist_case.train_counter]

    fig3 = plt.figure()
    plt.plot(hist_case.train_counter_epoch, hist_case.train_losses, color='blue')
    plt.scatter(hist_case.test_counter_epoch, hist_case.test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Epoch number')
    plt.ylabel('Cross Entropy loss value')
    plt.grid()
    fig3.savefig(results_dirname+name+'/loss_value_epoch.png')

    fig4 = plt.figure()
    plt.plot(hist_case.test_counter_epoch, hist_case.accuracy_hist_train, color='blue')
    plt.plot(hist_case.test_counter_epoch, hist_case.accuracy_hist_test, color='red')
    plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc='lower right')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy [%]')
    plt.grid()
    fig4.savefig(results_dirname+name+'/accuracy_epoch.png')




def train(epoch, net, optimizer, train_losses , train_counter, results_dirname, name, loss_choice='CE'):
  net.train()

  if not os.path.exists(results_dirname + name):
      os.makedirs(results_dirname + name)

  total_losses = 0
  data_seen_total = 0

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = net(data)

    if loss_choice == 'CE':
        loss = F.cross_entropy(output, target, size_average=False)
    elif loss_choice == 'L2_norm':
        target_vector = torch.zeros(len(output), 10)
        for target_num, target_value in enumerate(target):
            target_vector[target_num][target_value] = 1
        loss = F.mse_loss(output, target_vector, size_average=False)
    else:
        print('Invalid loss')

    total_losses += loss.item()
    data_seen_total += len(data)

    loss.backward()
    optimizer.step()




    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
      train_losses.append(total_losses / data_seen_total)

      total_losses = 0
      data_seen_total = 0
      torch.save(net.state_dict(), results_dirname+name+'/model.pth')
      torch.save(optimizer.state_dict(), results_dirname+name+'/optimizer.pth')


def test_validation(net, test_losses, accuracy_hist_test, loss_choice='CE'):

  net.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = net(data)

      if loss_choice == 'CE':
          test_loss += F.cross_entropy(output, target, size_average=False).item()
      elif loss_choice == 'L2_norm':
          target_vector = torch.zeros(len(output), 10)
          for target_num,target_value in enumerate(target):
              target_vector[target_num][target_value] = 1
          test_loss += F.mse_loss(output, target_vector, size_average=False).item()
      else:
          print('Invalid loss')
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  accuracy_hist_test.append(correct.item() * (100 / len(test_loader.dataset)))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def test_training(net, train_validation_losses, accuracy_hist_train, loss_choice='CE'):

  net.eval()
  train_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in train_loader:
      output = net(data)

      if loss_choice == 'CE':
          train_loss += F.cross_entropy(output, target, size_average=False).item()
      elif loss_choice == 'L2_norm':
          target_vector = torch.zeros(len(output), 10)
          for target_num,target_value in enumerate(target):
              target_vector[target_num][target_value] = 1
          train_loss += F.mse_loss(output, target_vector, size_average=False).item()
      else:
          print('Invalid loss')
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  train_loss /= len(train_loader.dataset)
  train_validation_losses.append(train_loss)
  accuracy_hist_train.append(correct.item() * (100 / len(train_loader.dataset)))
  print('\nTraining set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))


def perform_training_process(net, optimizer, n_epochs, hist_case, results_dirname, name_case, loss_choice='CE'):
    # initial validation
    test_validation(net=net, loss_choice=loss_choice, test_losses=hist_case.test_losses, accuracy_hist_test=hist_case.accuracy_hist_test)
    test_training(net=net, loss_choice=loss_choice, train_validation_losses=hist_case.train_validation_losses,
                  accuracy_hist_train=hist_case.accuracy_hist_train)

    # train and validate
    for epoch in range(1, n_epochs + 1):
        train(epoch=epoch, net=net, optimizer=optimizer, loss_choice=loss_choice, train_losses=hist_case.train_losses,
              train_counter=hist_case.train_counter, results_dirname=results_dirname, name=name_case)
        test_validation(net=net, loss_choice=loss_choice,  test_losses=hist_case.test_losses, accuracy_hist_test=hist_case.accuracy_hist_test)
        test_training(net=net, loss_choice=loss_choice,  train_validation_losses=hist_case.train_validation_losses,
                      accuracy_hist_train=hist_case.accuracy_hist_train)

    # print the final accuracy
    accuracy_out = open(results_dirname+name_case+"/acc.txt", "w")
    accuracy_out.write('\nFinal accuracy on validation: ' +
                       str(hist_case.accuracy_hist_test[len(hist_case.accuracy_hist_test) - 1]))
    accuracy_out.write('\nFinal accuracy on training: ' +
                       str(hist_case.accuracy_hist_train[len(hist_case.accuracy_hist_train) - 1]))
    accuracy_out.close()

    print('Classification accuracy on the test set:',
          hist_case.accuracy_hist_test[len(hist_case.accuracy_hist_test) - 1])
    print('Classification accuracy on the train set:',
          hist_case.accuracy_hist_train[len(hist_case.accuracy_hist_train) - 1])



# General parameters, equal for all
n_epochs = 8

### CASE 1 - LEARNING_RATE = 0.01, ERROR = CE
name_case = 'lr_0_01_net_1_CE_'
net = given_net_1()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# initialize the history - contains all the measurements taken during the training.
hist_case_1 = Train_History(n_epochs=n_epochs, train_dataset_length=len(train_loader.dataset))

perform_training_process(net=net, optimizer=optimizer, loss_choice='CE', n_epochs=n_epochs,
                         hist_case=hist_case_1, results_dirname=results_dirname, name_case=name_case)
plot_all_figures(hist_case_1, results_dirname=results_dirname, name=name_case)


### CASE 2 - LEARNING_RATE = 0.1, ERROR = CE
name_case = 'lr_0_1_net_1_CE_'
net = given_net_1()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)

# initialize the history - contains all the measurements taken during the training.
hist_case_2 = Train_History(n_epochs=n_epochs, train_dataset_length=len(train_loader.dataset))

perform_training_process(net=net, optimizer=optimizer, loss_choice='CE', n_epochs=n_epochs,
                         hist_case=hist_case_2, results_dirname=results_dirname, name_case=name_case)
plot_all_figures(hist_case_2, results_dirname=results_dirname, name=name_case)


### CASE 3 - LEARNING_RATE = 0.0001, ERROR = CE
name_case = 'lr_0_0001_net_1_CE_'
net = given_net_1()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.5)

# initialize the history - contains all the measurements taken during the training.
hist_case_3 = Train_History(n_epochs=n_epochs, train_dataset_length=len(train_loader.dataset))

perform_training_process(net=net, optimizer=optimizer, loss_choice='CE', n_epochs=n_epochs,
                         hist_case=hist_case_3, results_dirname=results_dirname, name_case=name_case)
plot_all_figures(hist_case_3, results_dirname=results_dirname, name=name_case)


### PLOT THE ACCURACY COMPARISON ON THE VALIDATION BETWEEN THE CASES 1,2,3


fig = plt.figure()
plt.plot(hist_case_1.test_counter_epoch, hist_case_1.accuracy_hist_test, color='blue')
plt.plot(hist_case_2.test_counter_epoch, hist_case_2.accuracy_hist_test, color='red')
plt.plot(hist_case_3.test_counter_epoch, hist_case_3.accuracy_hist_test, color='green')
plt.title('Accuracy comparison for different learning rate')
plt.legend(['0.01', '0.1', '0.0001'], loc='lower right')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy [%]')
plt.grid()
fig.savefig(results_dirname + 'comparison' + '/accuracy_LR_1_2_3.png')


### CASE 4 - Learning rate = 0.01, Loss func 0= L2_norm
name_case = 'lr_0_01_net_1_L2_norm_'
net = given_net_1()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# initialize the history - contains all the measurements taken during the training.
hist_case_4 = Train_History(n_epochs=n_epochs, train_dataset_length=len(train_loader.dataset))

perform_training_process(net=net, optimizer=optimizer, loss_choice='L2_norm', n_epochs=n_epochs,
                         hist_case=hist_case_4, results_dirname=results_dirname, name_case=name_case)
plot_all_figures(hist_case_4, results_dirname=results_dirname, name=name_case)


### PLOT THE ACCURACY COMPARISON ON THE VALIDATION BETWEEN THE CASES 1,4


fig = plt.figure()
plt.plot(hist_case_1.test_counter_epoch, hist_case_1.accuracy_hist_test, color='blue')
plt.plot(hist_case_4.test_counter_epoch, hist_case_4.accuracy_hist_test, color='red')
plt.title('Accuracy comparison for different loss functions, lr=0.1')
plt.legend(['Cross Entropy', 'L2 Norm'], loc='lower right')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy [%]')
plt.grid()
fig.savefig(results_dirname + 'comparison' + '/accuracy_1_4.png')


### CASE 5 - Network of other structure
name_case = 'lr_0_01_net_2_CE_'
net = given_net_2()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# initialize the history - contains all the measurements taken during the training.
hist_case_5 = Train_History(n_epochs=n_epochs, train_dataset_length=len(train_loader.dataset))

perform_training_process(net=net, optimizer=optimizer, loss_choice='CE', n_epochs=n_epochs,
                         hist_case=hist_case_5, results_dirname=results_dirname, name_case=name_case)
plot_all_figures(hist_case_5, results_dirname=results_dirname, name=name_case)



### PLOT THE ACCURACY COMPARISON ON THE VALIDATION BETWEEN THE CASES 1,5


fig = plt.figure()
plt.plot(hist_case_1.test_counter_epoch, hist_case_1.accuracy_hist_test, color='blue')
plt.plot(hist_case_5.test_counter_epoch, hist_case_5.accuracy_hist_test, color='red')
plt.title('Accuracy comparison for different net structure')
plt.legend(['Network 1', 'Network 2'], loc='lower right')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy [%]')
plt.grid()
fig.savefig(results_dirname + 'comparison' + '/accuracy_1_5.png')























