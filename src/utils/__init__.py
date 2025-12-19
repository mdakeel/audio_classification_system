import sys
import torch
import torch.nn as nn
from src.exception import CustomException
from torch.optim.lr_scheduler import StepLR
from src.constants import *

def accuracy(outputs, labels):
    """Calculate accuracy for a batch."""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_default_device():
    """Pick GPU if available, else CPU"""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def my_fit_method(epochs, lr, model, train_data_loader, val_loader, opt_func=torch.optim.SGD, grad_clip=GRAD_CLIP):
    """Training loop with OneCycleLR scheduler and gradient clipping."""
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, epochs=epochs, steps_per_epoch=len(train_data_loader)
    )
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []

        for batch in train_data_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate
            sched.step()
            
        # Validation Phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history, result

@torch.no_grad()
def evaluate(model, val_loader):
    """Evaluate model on validation set."""
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def predict_image(img, model, device):
    """
    Predict class for a single image.
    - Converts to batch of 1
    - Runs model
    - Returns class name
    """
    ximg = to_device(img.unsqueeze(0), device)
    yimg = model(ximg)

    # Debug: print probabilities
    probs = torch.nn.functional.softmax(yimg, dim=1)
    print("Prediction probabilities:", probs.cpu().detach().numpy())

    # Pick highest probability
    _, preds = torch.max(yimg, dim=1)
    class_names = ['cat', 'dog']
    return class_names[preds[0].item()]
