# %%
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt


torch.cuda.is_available()

# %%
celeba_train = torchvision.datasets.CelebA(root="data/", split="train", target_type='identity', download=True, transform=torchvision.transforms.ToTensor())
celeba_valid = torchvision.datasets.CelebA(root="data/", split="valid", target_type='identity', download=True, transform=torchvision.transforms.ToTensor())
# celeba_test = torchvision.datasets.CelebA(root="data/", split="test", download=True)

# %%
# Custom sampler class that generates batches with of triplets of images
class TripletSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.identity_array = self.create_identity_array(dataset)

    def __getitem__(self, i):
        # Get a random anchor image and its identity
        _, anchor_identity = self.dataset[i]
        # Get all indexes with the same identity
        same_identity_idx = self.identity_array[anchor_identity]
        # remove the anchor image from the list
        same_identity_idx = same_identity_idx[same_identity_idx != i]
        if len(same_identity_idx) == 0: return None
        # Get a random positive image
        positive_idx = same_identity_idx[np.random.randint(len(same_identity_idx))]
        # Get random index with different identity
        negative_idx = np.random.randint(len(self.dataset))
        while self.dataset.identity[negative_idx] == anchor_identity:
            negative_idx = np.random.randint(len(self.dataset))

        return (i, positive_idx, negative_idx)

    def __iter__(self):
        for i in range(len(self.dataset)):
            next_triple = self[i]
            if next_triple is None: continue
            yield next_triple

    def __len__(self):
        return len(self.dataset)

    def create_identity_array(self, dataset):
        # Create array where identity is the index to an array of all indexes of images of that identity
        identity_array = np.empty(max(dataset.identity)+1, dtype=object)
        for i, id in enumerate(dataset.identity):
            if identity_array[id] is None:
                identity_array[id] = np.array([i])
            else:
                identity_array[id] = np.append(identity_array[id], i)
                
        return identity_array

    def filter_identity_array(self, dataset, identity_array):
        # TODO
        for i, id in enumerate(dataset.identity):
            if id not in identity_array:
                identity_array[id] = None

        return identity_array




class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, triplet_idx):
        anchor_idx, positive_idx, negative_idx = triplet_idx
        a_img, a_label = self.dataset[anchor_idx]
        p_img, p_label = self.dataset[positive_idx]
        n_img, n_label = self.dataset[negative_idx]
        assert a_label == p_label and a_label != n_label
        return (a_img, p_img, n_img), (a_label.item(), p_label.item(), n_label.item())

    def __len__(self):
        return len(self.dataset)


triplet_sampler = TripletSampler(celeba_train)
triplet_dataset = TripletDataset(celeba_train)
triplet_data_loader = torch.utils.data.DataLoader(triplet_dataset,
                                                  batch_size=16,
                                                  #shuffle=False, # Should be True later
                                                  num_workers=0, # Needs to be 0 for me or calling iter never returns
                                                  sampler=triplet_sampler,
                                                  )
triplet_iter = iter(triplet_data_loader)

# %%
def plot_triplets(img_label_pairs):
    fig, ax = plt.subplots(len(img_label_pairs), 3, figsize=(10, 10))
    if len(img_label_pairs) == 1:
        ax = np.array([ax])
    titles = ["Anchor", "Positive", "Negative"]
    for i, (img, labels) in enumerate(img_label_pairs):
        for j in range(3):
            ax[i,j].imshow(img[j].permute(1, 2, 0))
            ax[i,j].set_title(f"{titles[j]}\nIdentity: {labels[j]}")
            ax[i,j].axis('off')
    
    # plt.subplots_adjust(hspace=1, wspace=-0.5)
    plt.tight_layout()
    plt.show()

# %%
example_iter = iter(range(len(triplet_data_loader)))

# %%
# A triplet batch is (Images, Labels) x (Anchor, Positive, Negative) x Batch Size x Channels x Height x Width
# = 2 x 3 x 16 x 3 x 218 x 178

def example(n=1):
    idx = triplet_sampler[next(example_iter)]
    triplet = triplet_dataset[idx]
    print("Triplet Batch size:")
    print(len(triplet), len(triplet[0]), "16", len(triplet[0][0]), *triplet[0][0][0].shape, sep=' x ')
    triplets = [triplet]

    for _ in range(n-1):
        idx = triplet_sampler[next(example_iter)]
        triplets.append(triplet_dataset[idx])
    
    plot_triplets(triplets)

example(3)

# %%
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity() # Override last layer with no impl 
        self.fc = nn.Linear(512, 1000) # Input features is 512 and output layers correspond to number of classes in ResNet, i.e. 1000 

    def forward(self, anchor, positive, negative):
        anchor = self.resnet(anchor)
        anchor = self.fc(anchor)
        positive = self.resnet(positive)
        positive = self.fc(positive)
        negative = self.resnet(negative)
        negative = self.fc(negative)
        return anchor, positive, negative


def train_triplet_net(triplet_net, triplet_data_loader, epochs=1, num_batches=None, lr=0.001, freeze=False):
    summary_steps = 1000

    if freeze:
        # freeze all layers except the last fc layer
        for param in triplet_net.resnet.parameters():
            param.requires_grad = False
        triplet_net.fc.requires_grad = True
    
    triplet_net.train()
    optimizer = optim.Adam(triplet_net.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    # TODO: Stop training with keyboard interrupt
    for epoch in range(epochs):
        running_loss = 0.0
        for i, triplet_batch in enumerate(triplet_data_loader):
            optimizer.zero_grad()
            # triplet_net.zero_grad()
            anchor, positive, negative = triplet_batch[0]
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            anchor, positive, negative = triplet_net(anchor, positive, negative)
            loss = criterion(anchor, positive, negative)
            loss.backward()
            # loss = triplet_net.loss(anchor, positive, negative)
            # loss.backward()
            optimizer.step()
            # triplet_net.optimizer.step()

            running_loss += loss.item()
            if i % summary_steps == summary_steps - 1:
                print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / summary_steps}")
                running_loss = 0.0
            if i == num_batches:
                print()
                break
        
    print("Finished Training\n")


def test_triplet_net(triplet_net, triplet_data_loader, batches=1, verbose=False):
    triplet_net.eval()
    with torch.no_grad():
        accuracies = []
        for i, triplet_batch in enumerate(triplet_data_loader):
            correct_predictions = 0
            anchor, positive, negative = triplet_batch[0]
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            anchor, positive, negative = triplet_net(anchor, positive, negative)
            anchor, positive, negative = anchor.cpu(), positive.cpu(), negative.cpu()
            for j in range(anchor.shape[0]):
                anchor_pos_dist = np.linalg.norm(anchor[j] - positive[j])
                anchor_neg_dist = np.linalg.norm(anchor[j] - negative[j])
                closest = np.argmin([anchor_neg_dist, anchor_pos_dist])
                correct_predictions += closest
                if verbose:
                    print("Anchor Positive Distance:", anchor_pos_dist)
                    print("Anchor Negative Distance:", anchor_neg_dist)
                    print("Closest:", ["Negative", "Positive"][closest])
                    print()
            accuracy = correct_predictions / anchor.shape[0]
            print(f"Batch: {i + 1}, Accuracy: {accuracy} ({correct_predictions}/{anchor.shape[0]})")
            accuracies.append(accuracy)

            if i == batches-1:
                print(f"\nTotal Accuracy: {sum(accuracies) / len(accuracies)}")
                break


# %%
triplet_net = TripletNet()
triplet_net.compile()
triplet_net.cuda()
comp_train = torch.compile(train_triplet_net)
comp_train(triplet_net, triplet_data_loader, epochs=1, lr=0.001, freeze=True)

