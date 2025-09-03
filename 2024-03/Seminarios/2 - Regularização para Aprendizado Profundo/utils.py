import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

# Set the seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize model, loss function, and optimizer
def model_builder(hidden_layers, hidden_neurons, dropout_rate):
    return lambda : MLP(input_size=28*28, 
                         num_hidden_layers=hidden_layers, 
                         hidden_size=hidden_neurons, 
                         output_size=10,
                         dropout_rate=dropout_rate)

# Define MLP model with customizable hidden layers, ability to use Dropout, and return embeddings
class MLP(nn.Module):
    def __init__(self, input_size=28*28, num_hidden_layers=2, hidden_size=[128], output_size=10, dropout_rate=0.0):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))  # First hidden layer
        for i in range(num_hidden_layers - 1):  # Additional hidden layers
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))  # Apply Dropout
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))

        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))  # Apply Dropout after the final hidden layer
        self.hidden_layers = nn.Sequential(*layers)  # Store hidden layers
        self.classifier = nn.Linear(hidden_size[-1], output_size)  # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image
        embeddings = self.hidden_layers(x)  # Get the output of the hidden layers (embeddings)
        output = self.classifier(embeddings)  # Classifier layer
        return embeddings, output  # Return embeddings and classification output

# Trainer class to handle training and validation
class Trainer:
    def __init__(self, model_builder, dataset, criterion, num_epochs=5, batch_size=64, k_folds=10, reg_type=None, reg_lambda=0.0, early_stopping=False, patience=5, seed=42, validation_split=0.2):
        self.model_builder = model_builder
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.reg_type = reg_type  # Regularization type: 'l1', 'l2', or None
        self.reg_lambda = reg_lambda  # Regularization strength
        self.early_stopping = early_stopping  # Enable or disable early stopping
        self.patience = patience  # Number of epochs to wait for improvement
        self.validation_split = validation_split  # Percentage of data for internal validation
        self.embeddings = []  # Store embeddings from validation data
        self.labels = []  # Store corresponding labels from validation data
        self.train_losses = []  # Store train losses for all folds
        self.val_losses = []  # Store validation losses for all folds
        self.weights_per_fold = []  # Store model weights after each fold
        self.seed = seed
        
        self.dataset = dataset

    def compute_regularization_loss(self):
        reg_loss = 0.0
        if self.reg_type == 'l1':
            for param in self.model.parameters():
                reg_loss += torch.sum(torch.abs(param))  # L1 regularization
        elif self.reg_type == 'l2':
            for param in self.model.parameters():
                reg_loss += torch.sum(param ** 2)  # L2 regularization
        return self.reg_lambda * reg_loss

    def train_fold(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            embeddings, outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Add regularization term to the loss
            reg_loss = self.compute_regularization_loss()
            loss += reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate_fold(self, valid_loader):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                embeddings, outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # Add regularization term to the loss
                reg_loss = self.compute_regularization_loss()
                loss += reg_loss
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Save embeddings and labels for t-SNE visualization
                self.embeddings.append(embeddings.cpu().numpy())
                self.labels.append(labels.cpu().numpy())

        accuracy = 100 * correct / total
        return running_loss / len(valid_loader), accuracy

    def save_model_weights(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.detach().cpu().numpy().flatten())  # Save model weights for analysis
        self.weights_per_fold.extend(weights)


    def split_data(self, train_ids):
        """
        Split the training data into subtraining and internal validation datasets.
        """
        train_size = int((1 - self.validation_split) * len(train_ids))
        subtrain_ids, internal_val_ids = random_split(train_ids, [train_size, len(train_ids) - train_size])
        return subtrain_ids, internal_val_ids

    def cross_validate(self):
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)  # Fix the random state for reproducibility

        for fold, (train_ids, valid_ids) in enumerate(kfold.split(self.dataset)):
            print(f'Fold {fold+1}/{self.k_folds}')

            # Split training data into subtraining and internal validation sets
            subtrain_ids, internal_val_ids = self.split_data(train_ids)

            # Create data loaders for the subtraining and internal validation datasets
            subtrain_loader = DataLoader(Subset(self.dataset, subtrain_ids), batch_size=self.batch_size, shuffle=True)
            internal_val_loader = DataLoader(Subset(self.dataset, internal_val_ids), batch_size=self.batch_size, shuffle=False)

            # Create data loader for the final validation dataset (held-out set)
            final_val_loader = DataLoader(Subset(self.dataset, valid_ids), batch_size=self.batch_size, shuffle=False)

            # Initialize model, optimizer, and loss function
            self.model = self.model_builder()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

            # Early stopping parameters
            best_loss = float('inf')
            patience_counter = 0

            fold_train_losses = []
            fold_val_losses = []

            # Training and validation for each epoch
            fold_val_losses.append(np.nan) # sync train and val loss arrays
            for epoch in range(self.num_epochs):
                train_loss = self.train_fold(subtrain_loader)
                internal_val_loss, _ = self.validate_fold(internal_val_loader)

                fold_train_losses.append(train_loss)
                fold_val_losses.append(internal_val_loss)

                print(f'Fold {fold+1}, Epoch [{epoch+1}/{self.num_epochs}], Subtrain Loss: {train_loss:.4f}, Internal Val Loss: {internal_val_loss:.4f}')

                # Early stopping logic
                if self.early_stopping:
                    if internal_val_loss < best_loss:
                        best_loss = internal_val_loss
                        patience_counter = 0  # Reset patience if the model improves
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f'Early stopping triggered in fold {fold+1} at epoch {epoch+1}')
                            break

            fold_train_losses.append(np.nan) # sync train and val loss arrays
            # Save the losses for this fold
            self.train_losses.append(fold_train_losses)
            self.val_losses.append(fold_val_losses)

            # Save the model weights for the current fold
            self.save_model_weights()

            # Evaluate the model on the held-out final validation set
            final_val_loss, final_val_accuracy = self.validate_fold(final_val_loader)
            print(f'Fold {fold+1}, Final Validation Loss: {final_val_loss:.4f}, Final Validation Accuracy: {final_val_accuracy:.2f}%')

    def plot_loss(self, ax):
        """
        Plot loss on the given ax.
        """
        for fold_idx in range(self.k_folds):
            ax.plot(self.train_losses[fold_idx], label=f'Fold {fold_idx+1} Train Loss', linestyle='--')
            ax.plot(self.val_losses[fold_idx], label=f'Fold {fold_idx+1} Val Loss', linestyle='-')

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Subtraining and Internal Validation Loss Across Folds')
        ax.legend(loc='upper right')

    def apply_tsne(self, ax, sample=100):
        """
        Apply t-SNE and plot the result on the given ax.
        """
        # Flatten the list of embeddings and labels
        all_embeddings = np.concatenate(self.embeddings, axis=0)[::sample]
        all_labels = np.concatenate(self.labels, axis=0)[::sample]

        # Apply t-SNE to reduce dimensionality to 2D
        tsne = TSNE(n_components=2, random_state=42)  # Fix the seed for reproducibility in t-SNE
        reduced_embeddings = tsne.fit_transform(all_embeddings)

        # Plot t-SNE result
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
        ax.set_title("t-SNE Visualization of MNIST Embeddings")
        plt.colorbar(scatter, ax=ax)

    def plot_weights_histogram(self, ax):
        """
        Plot histogram of weights after training for all folds on the given ax.
        """
        all_weights = np.concatenate(self.weights_per_fold, axis=0)  # Concatenate all weights from all folds
        ax.hist(all_weights, bins=50, alpha=0.75, color='blue')
        ax.set_title("Histogram of Weights Across All Folds")
        ax.set_xlabel("Weight values")
        ax.set_ylabel("Frequency")
