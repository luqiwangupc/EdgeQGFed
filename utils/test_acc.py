import torch
import torch.nn as nn
from datasets.SemiDatasets import ImgDataset
from models.GetModel import get_model
from omegaconf import OmegaConf
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils.test_evaluate import test_evaluate
from datasets.SemiDatasets import FedSemiDataset

class FULL_Model(nn.Module):
    """
    TSNE模型，结合了encoder和classifier，并且将classifier的最后一层分类层设置为Identity()
    """
    def __init__(self, encoder, classifier):
        super(FULL_Model, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        # self.classifier.model.classfier.fc3 = nn.Identity()
        self.classifier = nn.Identity()
        self.encoder.eval()
        self.classifier.eval()

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def run_model(config):
    device = torch.device(f"cuda:{config.train.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config.train.device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(0.5, 0.5)
    ])
    valset = FedSemiDataset(labeled_ratio=1, train=False, transform=transform, data_name=config.datasets.name)
    valloader = DataLoader(valset, batch_size=config.datasets.batch_size, shuffle=False, num_workers=config.datasets.num_workers)

    # 加载模型
    encoder = get_model(level=2, config=config)
    classifier = get_model(level=1, config=config)
    # 分开的结构
    state_dict = torch.load(
            str(os.path.join(config.train.ckpt_save_path, config.datasets.name, config.train.ckpt_save_name)),
            map_location='cpu')
    # state_dict = model.state_dict()
    classifier.load_state_dict(state_dict)      # 加载权重
    model = FULL_Model(encoder, classifier).to(device)

    # 模拟数据加载和模型预测过程
    val_metrics = test_evaluate(
        encoded_model=encoder,
        model=classifier,
        val_loader=valloader,
        device=device
    )
    print("\nValidation Metrics:")
    print(f"├── F1: {val_metrics['val_f1']:.4f}")
    print(f"└── Accuracy: {val_metrics['val_accuracy']:.2f}%")  # 添加百分号
    print(f"└── AUC_ROC: {val_metrics['val_auc']:.2f}")  # 添加百分号




if __name__ == '__main__':
    config = OmegaConf.load('config/formated_config.yaml')
    run_model(config)
