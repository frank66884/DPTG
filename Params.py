import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=4096, type=int, help='batch size')
	parser.add_argument('--adamw_lr', default=2e-3, type=float, help='learning rate for AdamW optimizer')
	parser.add_argument('--adamw_weight_decay', default=1e-4, type=float, help='weight decay for AdamW optimizer')
	parser.add_argument('--tstBat', default=8192, type=int, help='number of users in a testing batch')
	parser.add_argument('--epoch', default=9999, type=int, help='max number of epochs (training stops by early stopping)')
	parser.add_argument('--save', default='Dtem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--load', default=None, help='model name to load')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--block_num', default=2, type=int, help='number of hops in gcn precessing')
	parser.add_argument('--data', default='ml1m', type=str, help='name of dataset')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default='6', type=str, help='indicates which gpu to use')
	parser.add_argument('--dropout', default=0.2, type=float, help='Ratio of transformer layer dropout')
	parser.add_argument('--num_head', default=4, type=int, help='Multihead number of transformer layer')
	parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight decay')
	parser.add_argument('--patience', default=30, type=int, help='early stopping patience (number of test epochs without improvement)')
	parser.add_argument('--cl_rate', default=0, type=float, help='contrastive learning loss weight')
	parser.add_argument('--temp', default=0.2, type=float, help='temperature for InfoNCE contrastive loss')
	parser.add_argument('--eps', default=0.1, type=float, help='noise magnitude for SimGCL perturbation')
	parser.add_argument('--seq_maxlen', default=50, type=int, help='max sequence length for transformer channel')
	parser.add_argument('--mode', default='both', type=str, choices=['both', 'gnn_only', 'transformer_only'], help='channel mode: both | gnn_only | transformer_only')
	parser.add_argument('--cl_warmup', default=20, type=int, help='number of epochs before CL loss kicks in (linear warmup)')
	parser.add_argument('--intra_cl_rate', default=1e-3, type=float, help='intra-GNN contrastive learning loss weight (LPF vs HPF)')
	return parser.parse_args()
args = ParseArgs()

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 Main.py
# CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun --nproc_per_node=5 Main.py