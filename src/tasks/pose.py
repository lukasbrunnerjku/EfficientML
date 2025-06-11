from pathlib import Path
from omegaconf import OmegaConf
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from ..models.pose import build_model


class PoseTrainer:
    def __init__(self, cfg, model, optimizer, criterion, train_loader, val_loader):
        self.cfg = cfg
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.criterion,
        ) = self.accelerator.prepare(model, optimizer, train_loader, val_loader, criterion)
        
        self.log_dir = self.cfg.train.log_dir / str(time.strftime('%Y-%m-%d_%H-%M-%S'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.log_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(self.cfg, self.ckpt_dir / "cfg.yaml")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_val_loss = float("inf")
        self.current_epoch = 0
        self.global_step = 0
        
    def fit(self):
        for epoch in range(self.cfg.train.epochs):
            self.current_epoch = epoch
            train_loss = self._train_one_epoch()
            val_loss = self._validate()

            self.writer.add_scalar("train/avg_loss", train_loss, epoch)
            self.writer.add_scalar("val/avg_loss", val_loss, epoch)

            if self.accelerator.is_main_process:
                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint()
                    
    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for images, targets in self.train_loader:
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            
            if self.accelerator.is_main_process:
                if self.global_step % self.cfg.train.log_every:
                    self.writer.add_scalar("train/loss", loss, self.global_step)

            self.global_step += 1
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.inference_mode()
    def _validate(self):
        self.model.eval()
        running_loss = 0.0
        for images, targets in self.val_loader:
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def _save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.accelerator.get_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
        }
        torch.save(checkpoint, self.ckpt_dir / "best.ckpt")
    

class PoseCriterion(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        
    def forward(self, outputs, targets):
        return 
    
    
def main():
    cfg = OmegaConf.load("src/configs/pose.yaml")

    model = build_model(cfg.unet, cfg.head)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = PoseCriterion()
    train_loader, val_loader = build_loaders(cfg.data, cfg.train.batch_size)

    trainer = PoseTrainer(cfg, model, optimizer, criterion, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()