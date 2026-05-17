import argparse
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_lads import LADSDVS128Gesture
from smartrouterFrontend import SmartRouterFrontEnd
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, *args, **kwargs):
        super().__init__()
        self.front_end = SmartRouterFrontEnd(in_channels=3, hidden_channels=16, state_channels=3, H=128, W=128)

        conv = [
            layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(*args, **kwargs),
            layer.MaxPool2d(2, 2),
            layer.Dropout2d(0.2),
        ]

        for i in range(1, 5):
            conv.extend([
                layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spiking_neuron(*args, **kwargs),
                layer.MaxPool2d(2, 2),
            ])
            if i >= 3:
                conv.append(layer.Dropout2d(0.2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(*args, **kwargs),
            layer.Dropout(0.5),
            layer.Linear(512, 11),
            spiking_neuron(*args, **kwargs),
            layer.VotingLayer(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        outputs = []
        for t in range(T):
            S_t = self.front_end(x[t]).unsqueeze(0)
            outputs.append(self.conv_fc(S_t).squeeze(0))
        return torch.stack(outputs, dim=0).mean(dim=0)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="PyTorch SNN Training With Hybrid LADS")
    parser.add_argument("-T", default=16, type=int, help="time steps")
    parser.add_argument("-b", default=8, type=int, help="batch size")
    parser.add_argument("-epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", default=8, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("-data-dir", type=str, default="./data", help="root dir for DVS Gesture dataset")
    parser.add_argument("-out-dir", type=str, default="./logs", help="root dir for saving logs and checkpoint")
    parser.add_argument("-resume", type=str, help="resume from the checkpoint path")
    parser.add_argument("-amp", action="store_true", help="automatic mixed precision training")
    parser.add_argument("-cupy", action="store_true", help="use CuPy backend")
    parser.add_argument("-opt", type=str, default="adam", help="use adam or sgd optimizer")
    parser.add_argument("-lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("-momentum", type=float, default=0.9, help="momentum for sgd")
    parser.add_argument("-channels", type=int, default=128, help="channels of Conv2d in SNN")
    parser.add_argument("-device", type=str, default="cuda:0")
    parser.add_argument("-seed", type=int, default=2026, help="random seed")

    parser.add_argument("-lads-decay-func", type=str, default="er", help="LADS decay mode")
    parser.add_argument("-lads-decay-param", type=float, default=0.2, help="LADS decay parameter")
    parser.add_argument("-lads-patch-size", type=int, default=32, help="LADS patch size")
    parser.add_argument("-lads-min-decay", type=float, default=0.0, help="minimum decay clamp")
    parser.add_argument(
        "-lads-interpolate-patches",
        action="store_true",
        help="interpolate patch decay factors to full resolution",
    )
    parser.add_argument(
        "-split-by",
        type=str,
        default="number",
        choices=("number", "time"),
        help="event window splitting strategy",
    )

    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    lads_kwargs = {
        "decay_func": args.lads_decay_func,
        "decay_param": args.lads_decay_param,
        "patch_size": args.lads_patch_size,
        "interpolate_patches": args.lads_interpolate_patches,
        "min_decay": args.lads_min_decay,
        "ts_to_seconds_factor": 1.0,
        "device": "cpu",
    }

    train_set = LADSDVS128Gesture(
        root=args.data_dir,
        train=True,
        frames_number=args.T,
        split_by=args.split_by,
        lads_kwargs=lads_kwargs,
    )
    test_set = LADSDVS128Gesture(
        root=args.data_dir,
        train=False,
        frames_number=args.T,
        split_by=args.split_by,
        lads_kwargs=lads_kwargs,
    )

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
    )
    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
    )

    net = DVSGestureNet(
        channels=args.channels,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

    functional.set_step_mode(net, "m")
    if args.cupy:
        functional.set_backend(net, "cupy", instance=neuron.LIFNode)

    print(net)
    net.to(args.device)

    if args.opt == "adam":
        base_params = []
        front_end_params = []
        for name, param in net.named_parameters():
            if "front_end" in name:
                front_end_params.append(param)
            else:
                base_params.append(param)

        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": args.lr},
                {"params": front_end_params, "lr": args.lr * 5.0},
            ],
            lr=args.lr,
        )
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError(args.opt)

    scaler = amp.GradScaler(enabled=args.amp)

    out_dir = os.path.join(
        args.out_dir,
        f"T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}"
        f"_hybrid-lads-{args.lads_decay_func}_p{args.lads_patch_size}_{args.split_by}",
    )
    if args.amp:
        out_dir += "_amp"
    if args.cupy:
        out_dir += "_cupy"

    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    writer = SummaryWriter(out_dir)
    start_epoch = 0
    max_test_acc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        max_test_acc = checkpoint["max_test_acc"]
        print(f"Resuming from epoch {start_epoch}, max accuracy so far: {max_test_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        net.train()
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0

        for frame, label in train_data_loader:
            optimizer.zero_grad()
            functional.reset_net(net)
            frame = frame.to(args.device).transpose(0, 1)
            label = label.to(args.device)

            with amp.autocast(enabled=args.amp):
                out_fr = net(frame)
                label_onehot = F.one_hot(label, 11).float()
                loss = F.mse_loss(out_fr, label_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(dim=1) == label).sum().item()

        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_samples = 0

        with torch.no_grad():
            for frame, label in test_data_loader:
                functional.reset_net(net)
                frame = frame.to(args.device).transpose(0, 1)
                label = label.to(args.device)

                with amp.autocast(enabled=args.amp):
                    out_fr = net(frame)
                    label_onehot = F.one_hot(label, 11).float()
                    loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(dim=1) == label).sum().item()

        train_loss /= train_samples
        train_acc /= train_samples
        test_loss /= test_samples
        test_acc /= test_samples

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint = {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "max_test_acc": max_test_acc,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint_best.pth"))

        end_time = time.time()
        escape_time = datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"epoch = {epoch}, train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, "
            f"test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}, max_test_acc = {max_test_acc:.4f}"
        )
        print(
            f"train speed = {train_samples / (end_time - start_time):.4f} samples/s, "
            f"test speed = {test_samples / (end_time - start_time):.4f} samples/s"
        )
        print(f"escape time = {escape_time}")

    print(f"Max Test Accuracy: {max_test_acc}")
    writer.close()


if __name__ == "__main__":
    main()
