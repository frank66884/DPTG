import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

class DataHandler:
	def __init__(self):
		predir = 'Data/' + args.data + '/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		# mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor (放到当前进程对应的 GPU 上)
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse_coo_tensor(idxs, vals, shape).cuda(t.cuda.current_device())

	def loadUserSeq(self, num_users):
		"""Load user interaction sequences and build padded tensor + mask."""
		seq_file = self.predir + 'user_seq_train.pkl'
		with open(seq_file, 'rb') as f:
			user_seq_dict = pickle.load(f)  # dict: uid -> list of item ids

		seq_maxlen = args.seq_maxlen
		# Build padded sequences and masks for ALL users
		userSeqs = np.zeros((num_users, seq_maxlen), dtype=np.int64)
		userSeqMasks = np.ones((num_users, seq_maxlen), dtype=np.bool_)  # True = padding

		for uid in range(num_users):
			if uid in user_seq_dict:
				seq = user_seq_dict[uid]
				if len(seq) >= seq_maxlen:
					# Truncate to most recent seq_maxlen items
					seq = seq[-seq_maxlen:]
				# Right-align (left-pad with 0)
				start = seq_maxlen - len(seq)
				userSeqs[uid, start:] = seq
				userSeqMasks[uid, start:] = False  # non-padding positions
			# else: all zeros (padding), mask all True

		self.userSeqs = t.from_numpy(userSeqs).long().cuda(t.cuda.current_device())
		self.userSeqMasks = t.from_numpy(userSeqMasks).bool().cuda(t.cuda.current_device())

	def LoadData(self, distributed=False):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		# Load user sequences for the sequence channel
		self.loadUserSeq(args.user)

		trnData = TrnData(trnMat)
		if distributed:
			self.trnSampler = DistributedSampler(trnData, shuffle=True)
			self.trnLoader = data.DataLoader(trnData, batch_size=args.batch, sampler=self.trnSampler, num_workers=8,persistent_workers=True)
		else:
			self.trnSampler = None
			self.trnLoader = data.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=8,persistent_workers=True)
		tstData = TstData(tstMat, trnMat)
		if distributed:
			self.tstSampler = DistributedSampler(tstData, shuffle=False)
			self.tstLoader = data.DataLoader(tstData, batch_size=args.tstBat, sampler=self.tstSampler, num_workers=8,persistent_workers=True)
		else:
			self.tstSampler = None
			self.tstLoader = data.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=8,persistent_workers=True)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		# 全向量化负采样：批量生成 + 批量检测碰撞 + 循环修正残余
		n = len(self.rows)
		self.negs = np.random.randint(0, args.item, size=n).astype(np.int32)
		# 批量检测碰撞
		check = np.array([(self.rows[i], self.negs[i]) in self.dokmat for i in range(n)])
		while check.any():
			idx = np.where(check)[0]
			self.negs[idx] = np.random.randint(0, args.item, size=len(idx)).astype(np.int32)
			check[idx] = np.array([(self.rows[i], self.negs[i]) in self.dokmat for i in idx])

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])