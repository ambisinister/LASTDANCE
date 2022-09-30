import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import umap
import umap.plot
import matplotlib.pyplot as plt
import pickle

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

def get_model(modelname='resnet'):
    if modelname == 'alexnet':
        net = torchvision.models.alexnet(weights='IMAGENET1K_V1')
        net.classifier[6] = torch.nn.Linear(4096, 10)
    else:
        net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        net.fc = torch.nn.Linear(512,10)

    net.to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return net, criterion, optimizer

def train_net(net, trainloader, optimizer, criterion, epochs=40):

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
     
        print(f'[{epoch + 1}] loss: {running_loss}')

    print('Finished Training')

    torch.save(net.state_dict(), './weights.pth')
    return net

def test(net, testloader):

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to("cuda")
            labels = labels.to("cuda")

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def extract_features(feature_extractor, loader, node):
    feats = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to("cuda")
            feat = feature_extractor(images)[node]
            for b in feat:
                feats.append(b.cpu().numpy().flatten())

    return feats

def get_trajectories(net, testloader, testloader_SVHN, plot=False, single_node=None):

    # Create feature extractor: try to just do all of it and split if memory is an issue
    nodes, _ = get_graph_node_names(net)
    if single_node is not None:
        nodes = [nodes[single_node]]
        
    pairs = []
    try:
        svhn_labs = testset_SVHN.labels
    except:
        svhn_labs = testset_SVHN.dataset.labels[testset_SVHN.indices]
        
    labs = [*testset.targets, *[10 for _ in svhn_labs]]    

    for node in nodes:
        print(node)
        
        feature_extractor = create_feature_extractor(net, return_nodes=[node])

        feats_cifar = extract_features(feature_extractor, testloader, node)
        feats_svhn = extract_features(feature_extractor, testloader_SVHN, node)

        f = [*feats_cifar, *feats_svhn]

        if single_node is not None or node == nodes[-2]:
            pickle.dump([f, labs], open('./penult_features.pkl', 'wb'))

        mapper = umap.UMAP().fit(np.array(f))

        points = mapper.transform(np.array(f))
        #print(points)
        pairs.append(points)

        if plot:
            umap.plot.points(mapper, labels=np.array(labs))
            plt.savefig('./figs/{}.png'.format(node))

    pairs = [pt for pt in zip(*pairs)]
    pickle.dump([pairs, labs], open('./trajectories.pkl', 'wb'))

    return [pairs, labs]


        
if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((63,63)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    # CIFAR10
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #SVHN
    testset_SVHN = SVHN(root='./data', split='test',
                        download=True, transform=transform)
    testset_SVHN = torch.utils.data.Subset(testset_SVHN, list(range(10000)))
    testloader_SVHN = torch.utils.data.DataLoader(testset_SVHN, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)

    
    

    net, criterion, optimizer = get_model()

    try:
        net.load_state_dict(torch.load('./weights.pth'))
    except:
        net = train_net(net, trainloader, optimizer, criterion)

    # Set the model to eval
    # Comment this line out to leave batch norm layers in
    # net.eval()
    
    test(net, testloader)

    try:
        trajectories = pickle.load(open('./trajectories.pkl', 'rb'))
        penult = pickle.load(open('./penult_features.pkl', 'rb'))
    except:
        trajectories = get_trajectories(net, testloader, testloader_SVHN)
        penult = pickle.load(open('./penult_features.pkl', 'rb'))        
        
    X, y = trajectories
    X = np.array(X)

    mapper = umap.UMAP().fit(np.reshape(X, (X.shape[0], np.prod(X.shape[1:]))))
    umap.plot.points(mapper, labels=np.array(y))
    plt.show()

    P, Py = penult
    P = np.array(P)
    mapper = umap.UMAP().fit(np.reshape(P, (P.shape[0], np.prod(P.shape[1:]))))
    umap.plot.points(mapper, labels=np.array(Py))
    plt.show()
