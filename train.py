import argparse

import rich
import torch
import torch.nn.functional as F

from mingat.data import load_data
from mingat.model import GAT


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden', type=int, default=8)
    parser.add_argument('--nb_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.2)

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--dataset', type=str, choices=["wiki-cs", "cora"], default="cora")
    parser.add_argument('--epochs', type=int, default=800)

    args = parser.parse_args()
    rich.print(f"Training config: {args}")

    return args


def main():

    args = get_args()

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        model.eval()
        output = model(features, adj)
        acc_test = accuracy(output[idx_test], labels[idx_test])

        rich.print('epoch={:04d} | '.format(epoch + 1),
                   'loss_train={:.6f}'.format(loss_train.data.item()),
                   'acc_train={:.6f}'.format(acc_train.data.item()),
                   'loss_val={:.6f}'.format(loss_val.data.item()),
                   'acc_val={:.6f}'.format(acc_val.data.item()),
                   'acc_test={:.6f}'.format(acc_test.data.item()))


if __name__ == "__main__":
    main()