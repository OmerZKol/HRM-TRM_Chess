import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *

import torch
import torch.optim as optim
import torch.nn.functional as F

from .ChessNNet import ChessNNet as Cnet
sys.path.append('../..')

args = dotdict({
    'lr': 0.0001,  # Reduced learning rate for HRM transformer
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 32,  # Smaller batch size for better updates
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'warmup_epochs': 2,  # Learning rate warmup
})


class NNetWrapper():
    def __init__(self, game):
        self.nnet = Cnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        
        # Learning rate scheduler with warmup
        def get_lr_scale(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs  # Linear warmup
            else:
                return 1.0  # Constant after warmup
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            q_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v, out_q = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                l_q_halt = self.loss_q_halt(target_pis, out_pi, target_vs, out_v, out_q)
                l_q_continue = self.loss_q_continue(out_q)
                q_loss = 0.5 * (l_q_halt + l_q_continue)
                total_loss = l_pi + l_v + q_loss

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                q_losses.update(q_loss.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, Loss_q=q_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f'Learning rate: {current_lr:.6f}')

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        # Use HRM-compatible loss for log probabilities
        return -torch.mean(torch.sum(targets * outputs, dim=1))

    def loss_v(self, targets, outputs):
        # Use HRM-compatible loss for tanh values
        return torch.mean((targets - outputs.view(-1)) ** 2)
    
    def loss_q_halt(self, move_targets, move_preds, value_targets, value_preds, q_info):
        """
        Calculate Q-halt loss using the methodology from losses.py.
        Determines sequence correctness based on move and value predictions.
        """
        with torch.no_grad():
            # For Othello: move_targets is policy distribution, move_preds is log probabilities
            # Get the actual move from the target policy (argmax of the policy distribution)
            target_moves = torch.argmax(move_targets, dim=-1)
            predicted_moves = torch.argmax(move_preds, dim=-1)
            
            # Handle value predictions - convert to scalar if needed  
            if value_preds.dim() > 1:
                value_predictions = value_preds.squeeze(-1)
            else:
                value_predictions = value_preds
            
            # Move correctness: compare the predicted move with target move
            move_correct = (predicted_moves == target_moves)
            
            # Value correctness: sign agreement between target and prediction
            value_correct = torch.sign(value_targets) == torch.sign(value_predictions)
            
            # Overall sequence correctness (both move and value must be correct)
            seq_is_correct = move_correct & value_correct
            
        # Q-halt loss using binary cross-entropy with logits
        q_halt_logits = q_info["q_halt_logits"]
        if q_halt_logits.dim() > 1:
            q_halt_logits = q_halt_logits.squeeze(-1)
            
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, 
            seq_is_correct.to(q_halt_logits.dtype), 
            reduction="mean"
        )
        
        return q_halt_loss

    def loss_q_continue(self, q_info):
        # Simplified binary cross-entropy loss for continue logits
        if "target_q_continue" in q_info:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_info["q_continue_logits"],
                q_info["target_q_continue"],
                reduction="mean"
            )
            return q_continue_loss
        else:
            return torch.tensor(0.0, device=q_info["q_continue_logits"].device)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
