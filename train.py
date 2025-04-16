# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from mpl_toolkits.mplot3d.proj3d import transform
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data.config import cfg
from layers.modules import MultiBoxLoss
from data.widerface import WIDERDetection , detection_collate
from models.factory import build_net , basenet_factory
from models.enhancer import RetinexNet
from utils.DarkISP import Low_Illumination_Degrading
from PIL import Image
import inspect
from weights.pth_LoadLocalWeight import LoadLocalW
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser( description = 'DSFD face Detector Training With Pytorch' )
train_set = parser.add_mutually_exclusive_group()
parser.add_argument( '--batch_size' , default = 16 , type = int , help = 'Batch size for training' )
parser.add_argument( '--model' , default = 'dark' , type = str ,
                     choices = [ 'dark' , 'vgg' , 'resnet50' , 'resnet101' , 'resnet152' ] ,
                     help = 'model for training' )
parser.add_argument( '--resume' , default = None , type = str ,
                     help = 'Checkpoint state_dict file to resume training from' )
parser.add_argument( '--num_workers' , default = 1 , type = int , help = 'Number of workers used in dataloading' )
parser.add_argument( '--cuda' , default = True , type = bool , help = 'Use CUDA to train model' )
parser.add_argument( '--lr' , '--learning-rate' , default = 7e-4 , type = float , help = 'initial learning rate' )
parser.add_argument( '--momentum' , default = 0.9 , type = float , help = 'Momentum value for optim' )
parser.add_argument( '--weight_decay' , default = 5e-3 , type = float , help = 'Weight decay for SGD' )
parser.add_argument( '--gamma' , default = 0.1 , type = float , help = 'Gamma update for SGD' )
parser.add_argument( '--multigpu' , default = True , type = bool , help = 'Use mutil Gpu training' )
parser.add_argument( '--save_folder' , default = '../model/forDAINet/' ,
                     help = 'Directory for saving checkpoint models' )
parser.add_argument( '--local_rank' , type = int , help = 'local rank for dist' )

args = parser.parse_args()
global local_rank
local_rank = args.local_rank

if 'LOCAL_RANK' not in os.environ :
	os.environ[ 'LOCAL_RANK' ] = str( args.local_rank )

if torch.cuda.is_available() :
	if args.cuda :
		# torch.set_default_tensor_type('torch.cuda.FloatTensor')
		import torch.distributed as dist
		
		gpu_num = torch.cuda.device_count()
		if local_rank == 0 :
			print( 'Using {} gpus'.format( gpu_num ) )
		rank = int( os.environ[ 'RANK' ] )
		torch.cuda.set_device( rank % gpu_num )
		dist.init_process_group( 'nccl' )
	if not args.cuda :
		print(
			"WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed." )
		torch.set_default_tensor_type( 'torch.FloatTensor' )
else :
	torch.set_default_tensor_type( 'torch.FloatTensor' )

save_folder = os.path.join( args.save_folder , args.model )
if not os.path.exists( save_folder ) :
	os.mkdir( save_folder )

train_dataset = WIDERDetection( cfg.FACE.TRAIN_FILE , mode = 'train' )

val_dataset = WIDERDetection( cfg.FACE.VAL_FILE , mode = 'val' )
train_sampler = torch.utils.data.distributed.DistributedSampler( train_dataset , shuffle = True )
train_loader = data.DataLoader( train_dataset , batch_size = args.batch_size , num_workers = args.num_workers ,
                                collate_fn = detection_collate , sampler = train_sampler , pin_memory = True )
val_batchsize = args.batch_size
val_sampler = torch.utils.data.distributed.DistributedSampler( val_dataset , shuffle = True )
val_loader = data.DataLoader( val_dataset , batch_size = val_batchsize , num_workers = 0 ,
                              collate_fn = detection_collate , sampler = val_sampler , pin_memory = True )

min_loss = np.inf


def train() :
	per_epoch_size = len( train_dataset ) // (args.batch_size * torch.cuda.device_count())
	start_epoch = 0
	iteration = 0
	step_index = 0
	
	# 配置检测网络dsfd net
	basenet = basenet_factory( args.model )
	dsfd_net = build_net( 'train' , cfg.NUM_CLASSES , args.model )
	net = dsfd_net
	
	# 中断恢复
	if args.resume :
		if local_rank == 0 :
			print( 'Resuming training, loading {}...'.format( args.resume ) )
		start_epoch = net.load_weights( args.resume )
		iteration = start_epoch * per_epoch_size

	if not args.resume :
		if local_rank == 0 :
			print( 'Initializing weights...' )
		net.enhancement.apply( net.weights_init )
	
	# Scaling the lr
	# 设置了根据批次大小和gpu数量调整学习率的机制
	lr = args.lr * np.round( np.sqrt( args.batch_size / 4 * torch.cuda.device_count() ) , 4 )
	param_group = [ ]
	param_group += [ { 'params' : dsfd_net.enhancement.parameters() , 'lr' : lr } ]
	
	optimizer = optim.SGD( param_group , lr = lr , momentum = args.momentum , weight_decay = args.weight_decay )
	
	if args.cuda :
		if args.multigpu :
			# 采用数据并行模型，多gpu
			net = torch.nn.parallel.DistributedDataParallel( net.cuda() ,find_unused_parameters = True )  # net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda())
		# net = net.cuda()
		cudnn.benckmark = True
	
	criterion = MultiBoxLoss( cfg , args.cuda )  # 不占用显存
	
	# criterion_enhance = EnhanceLoss()
	if local_rank == 0 :
		print( 'Loading wider dataset...' )
		print( 'Using the specified args:' )
		print( args )
	
	for step in cfg.LR_STEPS :
		if iteration > step :
			step_index += 1
			adjust_learning_rate( optimizer , args.gamma , step_index )
	
	net.train()
	corr_mat = None
	
	trans = transforms.ToPILImage()
	
	for epoch in range( start_epoch , cfg.EPOCHES ) :
		losses = 0
		loss_l1 = 0
		loss_c1 = 0
		loss_l2 = 0
		loss_c2 = 0
		loss_mu = 0
		loss_en = 0
		
		if False:
			_epoch=epoch%40
			
			if _epoch == 0 :
				for param in net.module.backbone.parameters() :
					param.requires_grad = False
				net.module.backbone.eval()
				
				for name , param in net.module.enhancement.named_parameters() :
					if 'lap_pyramid' in name :
						param.requires_grad = False
					else :
						param.requires_grad = True
				net.module.enhancement.train()
				print('只训练增强模块')
			elif _epoch == 20 :
				for param in net.module.backbone.parameters() :
					param.requires_grad = True
				net.module.backbone.train()
				
				for param in net.module.enhancement.parameters() :
					param.requires_grad = False
				net.module.enhancement.eval()
				print('只训练检测模块')
			elif _epoch == 30:
				for param in net.module.backbone.parameters() :
					param.requires_grad = True
				net.module.backbone.train()
				
				for name , param in net.module.enhancement.named_parameters() :
					if 'lap_pyramid' in name :
						param.requires_grad = False
					else :
						param.requires_grad = True
				net.module.enhancement.train()
				print( '全训练' )
		
		# print(f"###epoch{epoch} is working")
		for batch_idx , (images , targets , _) in enumerate( train_loader ) :
			images = images.cuda() / 255.
			with torch.no_grad() :
				targetss = [ ann.cuda() for ann in targets ]
			img_dark = torch.empty(
				size = (images.shape[ 0 ] , images.shape[ 1 ] , images.shape[ 2 ] , images.shape[ 3 ]) ).cuda()
			# Generation of degraded data and AET groundtruth
			for i in range( images.shape[ 0 ] ) :
				img_dark[ i ] , _ = Low_Illumination_Degrading( images[ i ] , safe_invert = False )  # ISP方法生成低照度图像
				
				# print( '合成暗图' )
				# image = np.transpose( img_dark[ i ].detach().cpu().numpy() ,
				#                       (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
				# image = (image * 255).astype( np.uint8 )
				# plt.imshow( image )
				# plt.axis( 'off' )
				# plt.show()
			
			if iteration in cfg.LR_STEPS :
				step_index += 1
				adjust_learning_rate( optimizer , args.gamma , step_index )
			
			# 前向传播两个分支
			t0 = time.time()
			
			# print( '暗处理' )
			# image = np.transpose( img_dark[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
			# image = (image * 255).astype( np.uint8 )
			# plt.imshow( image )
			# plt.axis( 'off' )
			# plt.show()
			#
			# print( '正常' )
			# image = np.transpose( images[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
			# image = (image * 255).astype( np.uint8 )
			# plt.imshow( image )
			# plt.axis( 'off' )
			# plt.show()
			
			loss_enhance = net( x_dark = img_dark , x_light = images )
			# out = net( img_dark)
			
			# 损失函数整理
			
			loss_enhance.backward()
			optimizer.zero_grad()
			
			torch.nn.utils.clip_grad_norm_( net.parameters() , max_norm = 100 , norm_type = 2 )
			optimizer.step()
			t1 = time.time()
			loss_en += loss_enhance.item()
			
			# del loss
			# torch.cuda.empty_cache()  # 释放未使用的缓存内存
			
			if iteration % 100 == 0 :
				# 每次显示的损失只包含当前一个batch的平均损失
				tloss_en = loss_en / (batch_idx + 1)
				
				if local_rank == 0 :
					print( 'Timer: %.4f' % (t1 - t0) )
					print( 'epoch:' + repr( epoch ) + ' || iter:' + repr( iteration ))
					print( '->> enhanced loss:{:.4f}'.format(  tloss_en) )
					print( '->>lr:{}'.format( optimizer.param_groups[ 0 ][ 'lr' ] ) )
					# val( epoch , net , dsfd_net , criterion )
				# 这里应该不能清零，后面val有用  # losses = 0  # loss_l1 = 0  # loss_c1 = 0  # loss_l2 = 0  # loss_c2 = 0  # loss_mu = 0
			
			if iteration != 0 and iteration % 5000 == 0 :
				if local_rank == 0 :
					print( 'Saving state, iter:' , iteration )
					file = 'dsfd_' + repr( iteration ) + '.pth'
					torch.save( dsfd_net.state_dict() , os.path.join( save_folder , file ) )
			iteration += 1
		
		if (epoch + 1) >= 0 :
			val( epoch , net , dsfd_net , criterion )
		if iteration >= cfg.MAX_STEPS :
			break


def val( epoch , net , dsfd_net , criterion ) :
	net.eval()
	step = 0
	losses = torch.tensor( 0. ).cuda()
	# losses_enh = torch.tensor(0.).cuda()
	t1 = time.time()
	
	for batch_idx , (images , targets , img_paths) in enumerate( val_loader ) :
		if args.cuda :
			images = images.cuda() / 255.
			with torch.no_grad() :
				targets = [ ann.cuda() for ann in targets ]
		else :
			images = images / 255.
			with torch.no_grad() :
				targets = [ ann for ann in targets ]
		img_dark = torch.stack( [ Low_Illumination_Degrading( images[ i ] )[ 0 ] for i in range( images.shape[ 0 ] ) ] ,
		                        dim = 0 )
		loss_enhance = net.module.test_forward( x_dark = img_dark , x_light = images )
		
		loss =loss_enhance
		
		losses += loss.item()
		step += 1
	dist.reduce( losses , 0 , op = dist.ReduceOp.SUM )
	
	tloss = losses / step / torch.cuda.device_count()
	t2 = time.time()
	if local_rank == 0 :
		print( 'Timer: %.4f' % (t2 - t1) )
		print( 'test epoch:' + repr( epoch ) + ' || Loss:%.4f' % (tloss) )
	
	global min_loss
	if tloss < min_loss :
		if local_rank == 0 :
			print( 'Saving best state,epoch' , epoch )
			torch.save( dsfd_net.state_dict() , os.path.join( save_folder , 'dsfd.pth' ) )
		min_loss = tloss
	
	states = { 'epoch' : epoch , 'weight' : dsfd_net.state_dict() , }
	if local_rank == 0 :
		torch.save( states , os.path.join( save_folder , 'dsfd_checkpoint.pth' ) )


def adjust_learning_rate( optimizer , gamma , step ) :
	"""Sets the learning rate to the initial LR decayed by 10 at every
		specified step
	# Adapted from PyTorch Imagenet example:
	# https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	# lr = args.lr * args.batch_size / 4 * torch.cuda.device_count() * (gamma ** (step))
	for param_group in optimizer.param_groups :
		param_group[ 'lr' ] = param_group[ 'lr' ] * gamma


if __name__ == '__main__' :
	train()
