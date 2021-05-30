# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl


from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from guided_conv import SeparableGuidedConv2d


class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=32, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.gconv = SeparableGuidedConv2d(input_channels=3, output_channels=5, kernel_size=(3,3))
        self.cw_weights_gen = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3*9, kernel_size=(3,3), padding=(1,1)),
        )
        
        self.dw_weights_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.dw_weights_linear = nn.Linear(3, 3 * 5)

        self.l1 = nn.Linear(32 * 32 * 5, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 10)


    def forward(self, x):
        cw_weights = self.cw_weights_gen(x).reshape(-1, 3, 9, 32, 32)

        dw_weights = self.dw_weights_conv(x).squeeze()
        dw_weights = self.dw_weights_linear(dw_weights).reshape(-1, 3, 5)

        x = self.gconv(x, cw_weights, dw_weights)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = CIFAR10('.', train=True, download=True, transform=transforms.ToTensor())
    cifar10_test = CIFAR10('', train=False, download=True, transform=transforms.ToTensor())
    cifar10_train, cifar10_val = random_split(dataset, [40000, 10000])

    train_loader = DataLoader(cifar10_train, batch_size=args.batch_size)
    val_loader = DataLoader(cifar10_val, batch_size=args.batch_size)
    test_loader = DataLoader(cifar10_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
