import warnings
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')

import torch as t
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import DualChannelRec
from DataHandler import DataHandler
from Utils.Utils import pairPredict
import pickle
import os
import random
import numpy as np
import setproctitle
from tqdm import tqdm
import wandb

def seed_everything(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	t.manual_seed(seed)
	t.cuda.manual_seed(seed)
	t.cuda.manual_seed_all(seed)
	t.backends.cudnn.deterministic = True
	t.backends.cudnn.benchmark = False

class Coach:
	def __init__(self, handler, distributed=False, local_rank=0):
		self.handler = handler
		self.distributed = distributed
		self.local_rank = local_rank
		self.is_main = (local_rank == 0)

		if self.is_main:
			print('USER', args.user, 'ITEM', args.item)
			print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'clLoss', 'intraCLLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
		self.bestRecall = 0
		self.bestEpoch = 0
		self.noImproveCnt = 0  # early stopping 计数器

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		if self.is_main:
			log('Model Prepared')
		if args.load != None:
			self.loadModel()
			if '_best' in args.load:
				# 加载 best 模型时，截断 metrics 到 best epoch
				best_ep = self.bestEpoch
				num_tests = best_ep // args.tstEpoch + 1
				for key in self.metrics:
					self.metrics[key] = self.metrics[key][:num_tests]
				stloc = best_ep + 1
			else:
				stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
			if self.is_main:
				# 初始化 wandb (resumed) 并回放历史 metrics
				wandb.init(
					project='TransGNN',
					name=args.data + '_' + args.save,
					config=vars(args),
					resume='allow',
				)
				self._replayMetrics()
				log('Resumed from epoch %d' % stloc)
		else:
			stloc = 0
			if self.is_main:
				# 初始化 wandb
				wandb.init(
					project='TransGNN',
					name=args.data + '_' + args.save,
					config=vars(args),
				)
				log('Model Initialized')
		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch(ep)
			if self.is_main:
				log(self.makePrint('Train', ep, reses, tstFlag))
				wandb.log({
					'epoch': ep,
					'Train/Loss': reses['Loss'],
					'Train/BPR_Loss': reses['preLoss'],
					'Train/CL_Loss': reses['clLoss'],
					'Train/Intra_CL_Loss': reses['intraCLLoss'],
					'Train/LR': self.scheduler.get_last_lr()[0],
				}, step=ep)
			should_stop = t.tensor([0], dtype=t.long).cuda()
			if tstFlag:
				reses = self.testEpoch(ep)
				if self.is_main:
					log(self.makePrint('Test', ep, reses, tstFlag))
					wandb.log({
						'epoch': ep,
						'Test/Recall@%d' % args.topk: reses['Recall'],
						'Test/NDCG@%d' % args.topk: reses['NDCG'],
					}, step=ep)
					# Early Stopping 检查
					if reses['Recall'] > self.bestRecall:
						self.noImproveCnt = 0
						log('Recall improved!')
					else:
						self.noImproveCnt += 1
						log('No improvement: %d/%d' % (self.noImproveCnt, args.patience))
					self.saveHistory(reses, ep)
					if self.noImproveCnt >= args.patience:
						log('Early stopping at epoch %d (no improvement for %d test epochs)' % (ep, args.patience))
						should_stop.fill_(1)
				if self.distributed:
					dist.broadcast(should_stop, src=0)
			if should_stop.item() == 1:
				break
			self.scheduler.step()
			if self.is_main:
				print()
		if self.is_main:
			wandb.finish()

	def prepareModel(self):
		self.raw_model = DualChannelRec().cuda()
		if self.distributed:
			self.model = DDP(self.raw_model, device_ids=[self.local_rank], output_device=self.local_rank)
		else:
			self.model = self.raw_model
		self.opt = t.optim.AdamW(self.model.parameters(), lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=args.adamw_weight_decay)
		self.scheduler = t.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=args.patience, eta_min=args.lr * 0.01)
		self.scaler = t.amp.GradScaler('cuda')
	
	def trainEpoch(self, ep=0):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		if self.handler.trnSampler is not None:
			self.handler.trnSampler.set_epoch(ep)
		epLoss, epPreLoss, epCLLoss, epIntraCLLoss = 0, 0, 0, 0
		steps = len(trnLoader)
		self.model.train()
		with tqdm(enumerate(trnLoader), total=steps, desc='Epoch %d [Train]' % ep, ncols=100, disable=not self.is_main) as pbar:
			for i, tem in pbar:
				ancs, poss, negs = tem
				ancs = ancs.long().cuda()
				poss = poss.long().cuda()
				negs = negs.long().cuda()

				# Get user sequences for current batch
				user_seq = self.handler.userSeqs[ancs]      # (batch, seq_maxlen)
				seq_mask = self.handler.userSeqMasks[ancs]  # (batch, seq_maxlen)

				# 通过 DDP wrapper 调用 forward 以确保梯度同步 (AMP autocast)
				with t.amp.autocast('cuda'):
					outputs = self.model(self.handler.torchBiAdj, user_seq, seq_mask, ancs)
					if args.mode == 'both':
						user_embeds, item_embeds, gnn_view, seq_view, hpf_view = outputs
						bprLoss = self.raw_model.bprLoss(user_embeds, item_embeds, poss, negs)
						clLoss = self.raw_model.infoNCELoss(gnn_view, seq_view)
						intraCLLoss = self.raw_model.infoNCELoss(gnn_view, hpf_view)
						# CL warmup: linearly ramp from 0 to cl_rate over cl_warmup epochs
						cl_weight = args.cl_rate * min(1.0, ep / max(1, args.cl_warmup))
						intra_cl_weight = args.intra_cl_rate * min(1.0, ep / max(1, args.cl_warmup))
						loss = bprLoss + cl_weight * clLoss + intra_cl_weight * intraCLLoss
					elif args.mode == 'gnn_only':
						user_embeds, item_embeds, lpf_view, hpf_view = outputs
						bprLoss = self.raw_model.bprLoss(user_embeds, item_embeds, poss, negs)
						clLoss = t.tensor(0.0)
						intraCLLoss = self.raw_model.infoNCELoss(lpf_view, hpf_view)
						intra_cl_weight = args.intra_cl_rate * min(1.0, ep / max(1, args.cl_warmup))
						loss = bprLoss + intra_cl_weight * intraCLLoss
					else:
						user_embeds, item_embeds = outputs
						bprLoss = self.raw_model.bprLoss(user_embeds, item_embeds, poss, negs)
						clLoss = t.tensor(0.0)
						intraCLLoss = t.tensor(0.0)
						loss = bprLoss
	
				epLoss += loss.item()
				epPreLoss += bprLoss.item()
				epCLLoss += clLoss.item()
				epIntraCLLoss += intraCLLoss.item()
				self.opt.zero_grad()
				self.scaler.scale(loss).backward()
				self.scaler.unscale_(self.opt)
				t.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
				self.scaler.step(self.opt)
				self.scaler.update()
				if self.is_main:
					pbar.set_postfix(loss='%.4f' % loss.item())
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['preLoss'] = epPreLoss / steps
		ret['clLoss'] = epCLLoss / steps
		ret['intraCLLoss'] = epIntraCLLoss / steps
		return ret

	def testEpoch(self, ep=0):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg = 0, 0
		i = 0
		num = tstLoader.dataset.__len__()  # 全局测试用户数
		steps = len(tstLoader)

		# 预计算 discount 向量 (GPU)
		discount = 1.0 / t.log2(t.arange(2, args.topk + 2, dtype=t.float32).cuda())  # (topk,)
		cum_discount = t.cumsum(discount, dim=0)  # (topk,)

		self.model.eval()
		with t.no_grad(), t.amp.autocast('cuda'):
			# 只计算一次全量嵌入，避免每个 batch 重复前向传播
			usrEmbeds, itmEmbeds = self.raw_model.predict(
				self.handler.torchBiAdj,
				self.handler.userSeqs,
				self.handler.userSeqMasks
			)
			itmEmbedsT = t.transpose(itmEmbeds, 1, 0)  # (latdim, item) 预转置

			local_count = 0  # 本 rank 实际处理的用户数
			with tqdm(tstLoader, total=steps, desc='Epoch %d [Test] ' % ep, ncols=100, disable=not self.is_main) as pbar:
				for usr, trnMask in pbar:
					i += 1
					usr = usr.long().cuda()
					trnMask = trnMask.cuda()
					local_count += usr.shape[0]

					allPreds = t.mm(usrEmbeds[usr], itmEmbedsT) * (1 - trnMask) - trnMask * 1e8
					_, topLocs = t.topk(allPreds, args.topk)  # (batch, topk) on GPU
					recall, ndcg = self.calcRes(topLocs, self.handler.tstLoader.dataset.tstLocs, usr, discount, cum_discount)
					epRecall += recall
					epNdcg += ndcg
					if self.is_main:
						pbar.set_postfix(recall='%.4f' % (epRecall / local_count), ndcg='%.4f' % (epNdcg / local_count))

		# 分布式汇总
		if self.distributed:
			metrics_tensor = t.tensor([epRecall, epNdcg], dtype=t.float64).cuda()
			dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
			epRecall, epNdcg = metrics_tensor[0].item(), metrics_tensor[1].item()

		t.cuda.empty_cache()  # 释放测试阶段的 CUDA 缓存显存
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds, discount, cum_discount):
		"""GPU 批量计算 Recall 和 NDCG，避免 Python for 循环"""
		batch_size = topLocs.shape[0]
		topk = topLocs.shape[1]

		# 收集每个用户的测试物品并 padding（不可避免的 Python 循环，但只做数据整理）
		tst_lens = []
		max_tst_len = 0
		batch_tst = []
		for i in range(batch_size):
			tst = tstLocs[batIds[i]]
			batch_tst.append(tst)
			tst_lens.append(len(tst))
			if len(tst) > max_tst_len:
				max_tst_len = len(tst)

		# 构建 padding 后的测试物品张量 (GPU)
		tst_tensor = t.full((batch_size, max_tst_len), -1, dtype=t.long, device=topLocs.device)
		for i in range(batch_size):
			tst_tensor[i, :tst_lens[i]] = t.tensor(batch_tst[i], dtype=t.long, device=topLocs.device)
		tst_len_tensor = t.tensor(tst_lens, dtype=t.float32, device=topLocs.device)  # (batch,)

		# 广播比较: (batch, max_tst_len, 1) == (batch, 1, topk) → (batch, max_tst_len, topk)
		hits = (tst_tensor.unsqueeze(2) == topLocs.unsqueeze(1))

		# === Recall ===
		item_hit = hits.any(dim=2).float()  # (batch, max_tst_len)
		recall_per_user = item_hit.sum(dim=1) / tst_len_tensor  # (batch,)

		# === NDCG ===
		# 每个命中位置的 discount: hits * discount → (batch, max_tst_len, topk)
		hit_discount = hits.float() * discount.unsqueeze(0).unsqueeze(0)
		dcg_per_item = hit_discount.max(dim=2)[0]  # (batch, max_tst_len)
		dcg_per_user = dcg_per_item.sum(dim=1)  # (batch,)

		# ideal DCG
		ideal_len = t.minimum(tst_len_tensor, t.tensor(float(topk), device=topLocs.device)).long()
		max_dcg = cum_discount[ideal_len - 1]  # (batch,)
		ndcg_per_user = dcg_per_user / max_dcg  # (batch,)

		return recall_per_user.sum().item(), ndcg_per_user.sum().item()

	def _replayMetrics(self):
		"""将历史 metrics 回放到 wandb 中，保证曲线连续"""
		n = len(self.metrics['TrainLoss'])
		for i in range(n):
			ep = i * args.tstEpoch
			log_dict = {
				'epoch': ep,
				'Train/Loss': self.metrics['TrainLoss'][i],
				'Train/BPR_Loss': self.metrics['TrainpreLoss'][i],
				'Train/CL_Loss': self.metrics['TrainclLoss'][i],
			}
			if 'TrainintraCLLoss' in self.metrics and i < len(self.metrics['TrainintraCLLoss']):
				log_dict['Train/Intra_CL_Loss'] = self.metrics['TrainintraCLLoss'][i]
			if i < len(self.metrics['TestRecall']):
				log_dict['Test/Recall@%d' % args.topk] = self.metrics['TestRecall'][i]
			if i < len(self.metrics['TestNDCG']):
				log_dict['Test/NDCG@%d' % args.topk] = self.metrics['TestNDCG'][i]
			wandb.log(log_dict, step=ep)

	def saveHistory(self, testRes=None, ep=0):
		if args.epoch == 0:
			return
		os.makedirs('./History', exist_ok=True)
		os.makedirs('./Models', exist_ok=True)

		# 先更新 bestRecall，再写入历史文件
		isBest = testRes is not None and testRes['Recall'] > self.bestRecall
		if isBest:
			self.bestRecall = testRes['Recall']
			self.bestEpoch = ep

		his_content = {
			'metrics': self.metrics,
			'bestRecall': self.bestRecall,
			'bestEpoch': self.bestEpoch,
		}
		with open('./History/' + args.save + '.his', 'wb') as fs:
			pickle.dump(his_content, fs)

		content = {
			'model': self.raw_model,
			'optimizer': self.opt.state_dict(),
			'scheduler': self.scheduler.state_dict(),
			'scaler': self.scaler.state_dict(),
		}
		# 保存 latest 模型
		t.save(content, './Models/' + args.save + '_latest.mod')
		log('Latest Model Saved: %s_latest' % args.save)

		# 如果当前 Recall 是最优，保存 best 模型
		if isBest:
			t.save(content, './Models/' + args.save + '_best.mod')
			log('Best Model Saved: %s_best (Recall=%.4f)' % (args.save, self.bestRecall))

	def loadModel(self):
		model_path = './Models/' + args.load + '.mod'
		ckp = t.load(model_path, weights_only=False, map_location='cuda:%d' % self.local_rank)
		self.raw_model = ckp['model']
		if self.distributed:
			self.model = DDP(self.raw_model, device_ids=[self.local_rank], output_device=self.local_rank)
		else:
			self.model = self.raw_model
		self.opt = t.optim.AdamW(self.model.parameters(), lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=args.adamw_weight_decay)
		self.scheduler = t.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=args.patience, eta_min=args.lr * 0.01)
		self.scaler = t.amp.GradScaler('cuda')
		# 恢复 optimizer 和 scheduler 状态（兼容旧 checkpoint）
		if 'optimizer' in ckp:
			self.opt.load_state_dict(ckp['optimizer'])
			log('Optimizer state restored')
		if 'scheduler' in ckp:
			self.scheduler.load_state_dict(ckp['scheduler'])
			log('Scheduler state restored (lr=%.6f)' % self.scheduler.get_last_lr()[0])
		if 'scaler' in ckp:
			self.scaler.load_state_dict(ckp['scaler'])
			log('AMP GradScaler state restored')

		# history 文件名去掉 _best/_latest 后缀
		his_name = args.load.replace('_best', '').replace('_latest', '')
		with open('./History/' + his_name + '.his', 'rb') as fs:
			his_content = pickle.load(fs)
		# 兼容旧格式
		if isinstance(his_content, dict) and 'metrics' in his_content:
			self.metrics = his_content['metrics']
			self.bestRecall = his_content.get('bestRecall', 0)
		else:
			self.metrics = his_content
			self.bestRecall = 0
		# 兼容旧 checkpoint：补齐 intraCLLoss 指标
		for prefix in ('Train', 'Test'):
			key = prefix + 'intraCLLoss'
			if key not in self.metrics:
				self.metrics[key] = [0.0] * len(self.metrics.get(prefix + 'Loss', []))
		# 加载 bestEpoch，兼容旧格式
		if isinstance(his_content, dict) and 'bestEpoch' in his_content:
			self.bestEpoch = his_content['bestEpoch']
		elif self.metrics['TestRecall']:
			# 旧格式回退：从 metrics 中找最佳 recall 对应的 epoch
			best_idx = max(range(len(self.metrics['TestRecall'])), key=lambda i: self.metrics['TestRecall'][i])
			self.bestEpoch = best_idx * args.tstEpoch
		else:
			self.bestEpoch = 0
		log('Model Loaded from %s (bestRecall=%.4f, bestEpoch=%d)' % (model_path, self.bestRecall, self.bestEpoch))

if __name__ == '__main__':
	seed_everything(42)

	# 判断是否为分布式训练（通过 torchrun 启动时会设置 WORLD_SIZE）
	distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
	if distributed:
		local_rank = int(os.environ['LOCAL_RANK'])
		dist.init_process_group(backend='nccl')
		t.cuda.set_device(local_rank)
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
		local_rank = 0

	setproctitle.setproctitle('proc_title')
	logger.saveDefault = True

	if local_rank == 0:
		log('Start')
	handler = DataHandler()
	handler.LoadData(distributed=distributed)
	if local_rank == 0:
		log('Load Data')
	coach = Coach(handler, distributed=distributed, local_rank=local_rank)
	coach.run()

	if distributed:
		dist.destroy_process_group()