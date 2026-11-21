"""Loss functions: MultiSpectralLoss (training), VGGPerceptualLoss, ConfigurableMultiSpectralLoss (ablation)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vgg16

class MultiSpectralLoss(nn.Module):
    """Weighted sum of gradient, low-frequency, color-correlation and VGG perceptual losses."""
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, lambda_percep=0.1,
                 percep_layers=16, vgg_pretrained=True, device='cuda'):
        super(MultiSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_percep = lambda_percep
        self.percep_layers = percep_layers
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.vgg = vgg16(pretrained=vgg_pretrained).features[:percep_layers].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self, pred, target):
        grad_loss = self.gradient_loss(pred, target)
        low_freq_loss = self.low_frequency_loss(pred, target)
        color_corr_loss = self.color_correlation_loss(pred, target)
        percep_loss = F.l1_loss(self.vgg(pred), self.vgg(target))
        total_loss = (self.alpha * grad_loss + 
                     self.beta * low_freq_loss + 
                     self.gamma * color_corr_loss +
                     self.lambda_percep * percep_loss)
        return total_loss
    def gradient_loss(self, pred, target):
        gray_pred = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        gray_target = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        grad_pred_x = F.conv2d(gray_pred.unsqueeze(1), self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(gray_pred.unsqueeze(1), self.sobel_y, padding=1)
        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
        grad_target_x = F.conv2d(gray_target.unsqueeze(1), self.sobel_x, padding=1)
        grad_target_y = F.conv2d(gray_target.unsqueeze(1), self.sobel_y, padding=1)
        grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-6)
        return F.l1_loss(grad_pred, grad_target)
    def low_frequency_loss(self, pred, target):
        pred_low = F.avg_pool2d(pred, kernel_size=4, stride=4)
        target_low = F.avg_pool2d(target, kernel_size=4, stride=4)
        return F.mse_loss(pred_low, target_low)
    def color_correlation_loss(self, pred, target):
        pred_cc = torch.stack([
            pred[:, 0] * pred[:, 1],
            pred[:, 0] * pred[:, 2],
            pred[:, 1] * pred[:, 2],
            pred[:, 0] * pred[:, 0],
            pred[:, 1] * pred[:, 1],
            pred[:, 2] * pred[:, 2]
        ], dim=1)
        target_cc = torch.stack([
            target[:, 0] * target[:, 1],
            target[:, 0] * target[:, 2],
            target[:, 1] * target[:, 2],
            target[:, 0] * target[:, 0],
            target[:, 1] * target[:, 1],
            target[:, 2] * target[:, 2]
        ], dim=1)
        return F.l1_loss(pred_cc, target_cc)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        x_features = []
        y_features = []
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in [2, 7, 12, 21, 30]:
                x_features.append(x)
                y_features.append(y)
                
        loss = 0
        for i in range(len(x_features)):
            loss += self.weights[i] * self.criterion(x_features[i], y_features[i])
            
        return loss


class ConfigurableMultiSpectralLoss(nn.Module):
    def __init__(self, config, device='cuda'):
        super(ConfigurableMultiSpectralLoss, self).__init__()
        self.config = config
        self.components = config['components']
        self.weights = config['weights']
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        if self.components.get('percep_loss', False):
            self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        else:
            self.vgg = None
        
        self.device = device

    def forward(self, pred, target):
        total_loss = 0.0
        loss_components = {}
        
        if self.components.get('grad_loss', False):
            grad_loss = self.gradient_loss(pred, target)
            total_loss += self.weights.get('alpha', 0.5) * grad_loss
            loss_components['grad_loss'] = grad_loss.item()
        
        if self.components.get('low_freq_loss', False):
            low_freq_loss = self.low_frequency_loss(pred, target)
            total_loss += self.weights.get('beta', 0.3) * low_freq_loss
            loss_components['low_freq_loss'] = low_freq_loss.item()
        
        if self.components.get('color_corr_loss', False):
            color_corr_loss = self.color_correlation_loss(pred, target)
            total_loss += self.weights.get('gamma', 0.2) * color_corr_loss
            loss_components['color_corr_loss'] = color_corr_loss.item()
        
        if self.components.get('percep_loss', False) and self.vgg is not None:
            percep_loss = F.l1_loss(self.vgg(pred), self.vgg(target))
            total_loss += self.weights.get('lambda_percep', 0.1) * percep_loss
            loss_components['percep_loss'] = percep_loss.item()
        
        return total_loss, loss_components

    def gradient_loss(self, pred, target):
        gray_pred = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        gray_target = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        grad_pred_x = F.conv2d(gray_pred.unsqueeze(1), self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(gray_pred.unsqueeze(1), self.sobel_y, padding=1)
        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-6)
        
        grad_target_x = F.conv2d(gray_target.unsqueeze(1), self.sobel_x, padding=1)
        grad_target_y = F.conv2d(gray_target.unsqueeze(1), self.sobel_y, padding=1)
        grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-6)
        
        return F.l1_loss(grad_pred, grad_target)

    def low_frequency_loss(self, pred, target):
        pred_low = F.avg_pool2d(pred, kernel_size=4, stride=4)
        target_low = F.avg_pool2d(target, kernel_size=4, stride=4)
        return F.mse_loss(pred_low, target_low)

    def color_correlation_loss(self, pred, target):
        pred_cc = torch.stack([
            pred[:, 0] * pred[:, 1],
            pred[:, 0] * pred[:, 2],
            pred[:, 1] * pred[:, 2],
            pred[:, 0] * pred[:, 0],
            pred[:, 1] * pred[:, 1],
            pred[:, 2] * pred[:, 2]
        ], dim=1)
        
        target_cc = torch.stack([
            target[:, 0] * target[:, 1],
            target[:, 0] * target[:, 2],
            target[:, 1] * target[:, 2],
            target[:, 0] * target[:, 0],
            target[:, 1] * target[:, 1],
            target[:, 2] * target[:, 2]
        ], dim=1)
        
        return F.l1_loss(pred_cc, target_cc)


def build_loss(config, device='cuda'):
    return ConfigurableMultiSpectralLoss(config, device)