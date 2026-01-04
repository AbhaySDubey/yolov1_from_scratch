import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CNNBlock, Yolov1
from dataset import VOCDataset
from utils import intersection_over_union, non_max_suppresssion, mean_average_precision, cellboxes_to_bboxes, get_bboxes, plot_image, save_checkpoint, load_checkpoint
from loss import YoloLoss
from datetime import datetime
import os


seed = 123
torch.manual_seed(seed)

# hyperparameters
learning_rate = 0.00002
device = "cuda" if torch.cuda.is_available() else "cpu"
# the batch-size in the paper is 64, given my potato machine 8 seems reasonable enough
batch_size = 8
# for now, this is set to 0, i'll change it later
weight_decay = 0
epochs = 100
num_workers = 2
pin_memory = True
load_model = False
load_model_file = "sample_model.pth"
img_dir = "dataset/images"
label_dir = "dataset/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, bboxes):
        for transf in self.transforms:
            img, bboxes = transf(img), bboxes
        return img, bboxes
            
transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])


def trainer(train_loader, model, optimizer, loss_func):
    # does one training loop through the entire dataset
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_func(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
        
    print(f"Mean loss: {sum(mean_loss)/len(mean_loss)}")

def main():
    curr_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    model_checkpoint_save_dir = 'logs/training_'+curr_datetime
    print(model_checkpoint_save_dir)
    
    if not os.path.exists(model_checkpoint_save_dir):
        os.makedirs(model_checkpoint_save_dir)
    
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_func = YoloLoss()
    
    if load_model:
        load_checkpoint(torch.load(load_model_file), model=model, optimizer=optimizer)
    
    # tranform = tr
    train_dataset = VOCDataset(
        "dataset/100examples.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    
    val_dataset = VOCDataset(
        "dataset/test.csv",
        transform=transform,
        img_dir=img_dir,
        label_dir=label_dir
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False
    )
    best_mAP = 0.0
    
    for epoch in range(epochs):
        trainer(train_loader=train_loader, model=model, optimizer=optimizer, loss_func=loss_func)
        
        if epoch%10 == 0 or epoch==epochs-1:
            model.eval()
            with torch.no_grad():
                pred_bboxes, true_bboxes = get_bboxes(
                    val_loader, model, iou_threshold=0.5, threshold=0.4
                )
                mAP = mean_average_precision(
                    pred_boxes=pred_bboxes,
                    true_boxes=true_bboxes,
                    iou_threshold=0.5,
                    box_format="midpoint",
                    num_classes=20
                )
                print(f"Validation mAP: {mAP}")
            
            model.train()
            
            model_checkpoint_fn = os.path.join(model_checkpoint_save_dir, f'epoch_{epoch}.pth')

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mAP": mAP,
            }
            
            if mAP > best_mAP:
                best_mAP = mAP
                model_checkpoint_fn = os.path.join(model_checkpoint_save_dir, 'best.pth')
                state['best_mAP'] = best_mAP
            
            save_checkpoint(state=state, filename=model_checkpoint_fn)
            



if __name__ == "__main__":
    main()
