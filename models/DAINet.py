# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable , Function
from torchvision import transforms

from layers import *
from data.config import cfg
from layers.modules.enhance_loss import EnhanceLoss,smooth
import cv2
import inspect

import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure as ssim


class Interpolate( nn.Module ) :
	# 插值的方法对张量进行上采样或下采样
	def __init__( self , scale_factor ) :
		super( Interpolate , self ).__init__()
		self.scale_factor = scale_factor
	
	def forward( self , x ) :
		x = nn.functional.interpolate( x , scale_factor = self.scale_factor , mode = 'nearest' )
		return x


class FEM( nn.Module ) :
	"""docstring for FEM"""
	
	def __init__( self , in_planes ) :
		super( FEM , self ).__init__()
		inter_planes = in_planes // 3
		inter_planes1 = in_planes - 2 * inter_planes
		self.branch1 = nn.Conv2d( in_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 )
		
		self.branch2 = nn.Sequential(
				nn.Conv2d( in_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) ,
				nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes , inter_planes , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) )
		self.branch3 = nn.Sequential(
				nn.Conv2d( in_planes , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) ,
				nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes1 , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) ,
				nn.ReLU( inplace = True ) ,
				nn.Conv2d( inter_planes1 , inter_planes1 , kernel_size = 3 , stride = 1 , padding = 3 , dilation = 3 ) )
	
	def forward( self , x ) :
		x1 = self.branch1( x )
		x2 = self.branch2( x )
		x3 = self.branch3( x )
		out = torch.cat( (x1 , x2 , x3) , dim = 1 )
		out = F.relu( out , inplace = True )
		return out


# 仿射变换
class SFT_layer( nn.Module ) :
	def __init__( self , in_ch = 3 , inter_ch = 32 , out_ch = 3 , kernel_size = 3 ) :
		super().__init__()
		self.encoder = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) ,
		                              nn.LeakyReLU( True ) , )
		self.decoder = nn.Sequential( nn.Conv2d( inter_ch , out_ch , kernel_size , padding = kernel_size // 2 ) )
		self.shift_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )
		self.scale_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )
	
	def forward( self , x , guide ) :
		x = self.encoder( x )
		scale = self.scale_conv( guide )
		shift = self.shift_conv( guide )
		x = x + x * scale + shift
		x = self.decoder( x )
		return x


# DEM
class Trans_high( nn.Module ) :
	def __init__( self , in_ch = 3 , inter_ch = 32 , out_ch = 3 , kernel_size = 3 ) :
		super().__init__()
		
		self.encoder = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) ,
		                              nn.LeakyReLU( True ) , )
		self.decoder = nn.Sequential( nn.Conv2d( inter_ch , out_ch , kernel_size , padding = kernel_size // 2 ) )
		self.shift_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )
		self.scale_conv = nn.Sequential( nn.Conv2d( in_ch , inter_ch , kernel_size , padding = kernel_size // 2 ) )
	
	def forward( self , x , guide ) :
		_x = self.encoder( x )
		scale = self.scale_conv( guide )
		shift = self.shift_conv( guide )
		_x = _x + _x * scale + shift
		_x = self.decoder( _x )
		
		return x + _x


# 上采样
class Up_guide( nn.Module ) :
	def __init__( self , kernel_size = 1 , ch = 3 ) :
		super().__init__()
		self.up = nn.Sequential( nn.Upsample( scale_factor = 2 , mode = "bilinear" , align_corners = True ) ,
		                         # 这里的卷积文章中并没有提到，AI认为可以确保引导信息与高频分量的特征在空间和语义上对齐，我持怀疑态度
		                         nn.Conv2d( ch , ch , kernel_size , stride = 1 , padding = kernel_size // 2 ,
		                                    bias = False ) )
	
	def forward( self , x ) :
		return self.up( x )


# 拉普拉斯金字塔
class Lap_Pyramid_Conv( nn.Module ) :
	def __init__( self , num_high = 3 , kernel_size = 5 , channels = 3 ) :
		super().__init__()
		
		self.num_high = num_high
		self.kernel = self.gauss_kernel( kernel_size , channels )
	
	def gauss_kernel( self , kernel_size , channels ) :
		kernel = cv2.getGaussianKernel( kernel_size , 0 ).dot( cv2.getGaussianKernel( kernel_size , 0 ).T )
		kernel = torch.FloatTensor( kernel ).unsqueeze( 0 ).repeat( channels , 1 , 1 , 1 )
		kernel = torch.nn.Parameter( data = kernel , requires_grad = False )
		return kernel
	
	def conv_gauss( self , x , kernel ) :
		n_channels , _ , kw , kh = kernel.shape
		x = torch.nn.functional.pad( x , (kw // 2 , kh // 2 , kw // 2 , kh // 2) ,
		                             mode = 'reflect' )  # replicate    # reflect
		x = torch.nn.functional.conv2d( x , kernel , groups = n_channels )
		return x
	
	def downsample( self , x ) :
		return x[ : , : , : :2 , : :2 ]
	
	def pyramid_down( self , x ) :
		return self.downsample( self.conv_gauss( x , self.kernel ) )
	
	def upsample( self , x ) :
		up = torch.zeros( (x.size( 0 ) , x.size( 1 ) , x.size( 2 ) * 2 , x.size( 3 ) * 2) , device = x.device )
		up[ : , : , : :2 , : :2 ] = x * 4
		
		return self.conv_gauss( up , self.kernel )
	
	def pyramid_decom( self , img ) :
		self.kernel = self.kernel.to( img.device )
		current = img
		pyr = [ ]
		for _ in range( self.num_high ) :
			down = self.pyramid_down( current )
			up = self.upsample( down )
			diff = current - up
			pyr.append( diff )
			current = down
		
		pyr.append( current )
		return pyr
	
	def pyramid_recons( self , pyr ) :
		image = pyr[ 0 ]
		for level in pyr[ 1 : ] :
			up = self.upsample( image )
			image = up + level
		return image


class SpatialAttention( nn.Module ) :
	def __init__( self , kernel_size = 5 ) :
		super().__init__()
		assert kernel_size in (3 , 5 , 7) , "kernel size must be 3 or 5 or 7"
		
		self.conv = nn.Conv2d( 2 , 1 , kernel_size , padding = kernel_size // 2 , bias = False )
		self.sigmoid = nn.Sigmoid()
	
	def forward( self , x ) :
		avgout = torch.mean( x , dim = 1 , keepdim = True )
		maxout , _ = torch.max( x , dim = 1 , keepdim = True )
		attention = torch.cat( [ avgout , maxout ] , dim = 1 )
		attention = self.conv( attention )
		return self.sigmoid( attention ) * x


# CGM
class Trans_guide( nn.Module ) :
	# 这里和论文中所说的通道为32有出入
	def __init__( self , ch = 16 ) :
		super().__init__()
		
		self.layer = nn.Sequential( nn.Conv2d( 6 , ch , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
		                            SpatialAttention( 3 ) , nn.Conv2d( ch , 3 , 3 , padding = 1 ) , )
	
	def forward( self , x ) :
		return self.layer( x )


# L通道
class Trans_low( nn.Module ) :
	def __init__( self , ch_blocks = 64 , ch_mask = 16 , ) :
		super().__init__()
		
		self.encoder = nn.Sequential( nn.Conv2d( 3 , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
		                              nn.Conv2d( 16 , ch_blocks , 3 , padding = 1 ) , nn.LeakyReLU( True ) )
		
		self.mm1 = nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 1 , padding = 0 )
		self.mm2 = nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 3 , padding = 3 // 2 )
		self.mm3 = nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 5 , padding = 5 // 2 )
		self.mm4 = nn.Conv2d( ch_blocks , ch_blocks // 4 , kernel_size = 7 , padding = 7 // 2 )
		
		self.decoder = nn.Sequential( nn.Conv2d( ch_blocks , 16 , 3 , padding = 1 ) , nn.LeakyReLU( True ) ,
		                              nn.Conv2d( 16 , 3 , 3 , padding = 1 ) )
		
		self.trans_guide = Trans_guide( ch_mask )
	
	def forward( self , x ) :
		x1 = self.encoder( x )
		# 这里是不是应该对应mm1-4
		x1_1 = self.mm1( x1 )
		x1_2 = self.mm1( x1 )
		x1_3 = self.mm1( x1 )
		x1_4 = self.mm1( x1 )
		x1 = torch.cat( [ x1_1 , x1_2 , x1_3 , x1_4 ] , dim = 1 )
		x1 = self.decoder( x1 )
		
		out = x + x1
		out = torch.relu( out )  # 当前通道输出
		
		mask = self.trans_guide( torch.cat( [ x , out ] , dim = 1 ) )  # 输出给high-layer
		return out , mask


class illumination_global( nn.Module ) :
	# 让亮图前后LF趋同，暗图通过后LF和亮图的LF也趋同
	# 计划用神经网络结合isp方法（iayolo）来实现
	def __init__( self , kernel_size = 7 ) :
		super().__init__()
		# assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1
		self.downsample = nn.Sequential( nn.Conv2d( 3 , 3 , kernel_size , padding = padding , stride = 2 ) ,
		                                 nn.ReLU( True ) , )
		self.encoder = nn.Sequential( nn.Conv2d( 3 , 3 , kernel_size , padding = padding ) , nn.ReLU( True ) ,
		                              nn.Conv2d( 3 , 32 , kernel_size , padding = padding ) , nn.LeakyReLU( True ) )
		
		self.decoder = nn.Sequential( nn.Conv2d( 32 , 3 , kernel_size , padding = kernel_size // 2 ) )
	
	def forward( self , img , LF , level ) :
		# 残差机制
		for i in range( level ) :  # LF HF3 HF2 HF1 0 1 2 3
			img = self.downsample( img )
		# 提取光照特征
		scale = self.encoder( LF )
		scale = self.decoder( scale )
		LF = LF * scale + LF
		
		return LF


# class DIPmodule():


class down_finetune( nn.Module ) :
	# 空间注意力机制 这里可能会出现色彩的问题，应该对通道也设置注意力
	def __init__( self , kernel_size = 3 , channels = 3 ) :
		super().__init__()
		self.conv = nn.Sequential( nn.Conv2d( 2 , 1 , kernel_size , padding = 1 ) , nn.Sigmoid() )
		self.downsample = nn.Sequential( nn.Conv2d( channels , channels , kernel_size , padding = 1 , stride = 2 ) ,
		                                 nn.Sigmoid() )
	
	def forward( self , HF , level , LF ) :
		avg_out = torch.mean( HF , dim = 1 , keepdim = True )
		max_out , _ = torch.max( HF , dim = 1 , keepdim = True )
		attention = torch.cat( [ avg_out , max_out ] , dim = 1 )
		
		HF = (self.conv( attention ) * HF) + HF
		for i in range( level ) :  # LF HF3 HF2 HF1 0 1 2 3
			HF = self.downsample( HF )
		out = LF + HF
		
		return out


class DENet( nn.Module ) :
	def __init__( self , num_high = 5 , ch_blocks = 32 , up_ksize = 1 , high_ch = 32 , high_ksize = 3 , ch_mask = 32 ,
	              gauss_kernel = 5 ) :
		super().__init__()
		self.mseloss=EnhanceLoss()
		self.num_high = num_high
		self.lap_pyramid = Lap_Pyramid_Conv( num_high , gauss_kernel )
		self.trans_low = Trans_low( ch_blocks , ch_mask )
		
		# 这里的setattr和后面的getattr是联动的
		for i in range( 0 , self.num_high ) :
			self.__setattr__( 'up_guide_layer_{}'.format( i ) , Up_guide( up_ksize , ch = 3 ) )
			self.__setattr__( 'trans_high_layer_{}'.format( i ) , Trans_high( 3 , high_ch , 3 , high_ksize ) )
	
	# (HF_d + LF_l) VS (HF_l + LF_l) 增强模块提取高频信号效果好；输出图像是亮度良好的。
	def forward( self , x_dark , x_light ) :
		
		# image = np.transpose( x_dark[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )
		# # 保存图像到文件
		# plt.savefig( f'暗化图.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		
		# HF1 HF2 HF3 LF
		pyrs_d = self.lap_pyramid.pyramid_decom( img = x_dark )
		pyrs_l = self.lap_pyramid.pyramid_decom( img = x_light )
		
		# for i in range( len(pyrs_d) ) :
		# 	image = np.transpose( pyrs_d[i][ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# 	image = (image * 255).astype( np.uint8 )
		# 	plt.imshow( image )
		# 	plt.axis( 'off' )
		# 	# 保存图像到文件
		# 	plt.savefig( f'HF_d_{ i}.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		
		trans_pyrs = [ ]  # 被处理的金字塔（主通路暗图）
		trans_pyrs_l = [ ]
		reconLL_pyrs = [ ]  # 重组金字塔 HF_l LF_l
		reconDL_pyrs = [ ]  # 重组金字塔 HF_d LF_l（原始LF）
		reconLLe_pyrs = [ ]  # 重组金字塔 HF_l LF_l(亮度增强)
		reconLD_pyrs = [ ]
		
		# trans_LF = self.illum_global( img = x_dark , LF = pyrs_d[ -1 ] , level = self.num_high )
		# debug_LF = self.illum_global( img = x_light , LF = pyrs_l[ -1 ] , level = self.num_high )
		
		trans_LF,guide = self.trans_low( pyrs_d[ -1 ] )
		trans_LF_l,guide_l = self.trans_low( pyrs_l[ -1 ] )
		# loss_LFenh = F.l1_loss( trans_LF , pyrs_d[ -1 ].detach() ) + (1. - ssim( trans_LF , pyrs_d[ -1 ].detach() ))
		trans_pyrs.append( pyrs_d[ -1 ] )
		trans_pyrs_l.append( pyrs_l[ -1 ])
		
		# image = np.transpose( trans_LF[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )
		# # 保存图像到文件
		# plt.savefig( f'LF_enh.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		
		commom_guide = [ ]
		commom_guide_l = [ ]
		for i in range( self.num_high ) :
			guide = self.__getattr__( 'up_guide_layer_{}'.format( i ) )( guide )
			commom_guide.append( guide )
			
			guide_l = self.__getattr__( 'up_guide_layer_{}'.format( i ) )( guide_l )
			commom_guide_l.append( guide_l )
		
		# image = np.transpose( guide[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )
		# # 保存图像到文件
		# plt.savefig( f'common_guide_{i}.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )

		loss_HFenh = 0
		loss_HFenh_l = 0
		for i in range( self.num_high ) :
			trans_HF = self.__getattr__( 'trans_high_layer_{}'.format( i ) )(
					pyrs_d[ -2 - i ] , commom_guide[ i ] )
			trans_pyrs.append( trans_HF )  # LF HF3 HF2 HF1
			loss_HFenh +=F.l1_loss(trans_HF,pyrs_d[-2-i].detach()) + (1. - ssim(trans_HF,pyrs_d[-2-i].detach()))
			
			trans_HF_l = self.__getattr__( 'trans_high_layer_{}'.format( i ) )(
					pyrs_l[ -2 - i ] , commom_guide_l[ i ] )
			trans_pyrs_l.append( trans_HF_l )
			loss_HFenh_l +=F.l1_loss(trans_HF_l,pyrs_l[-2-i].detach()) + (1. - ssim(trans_HF_l,pyrs_l[-2-i].detach()))
			
			# image = np.transpose( trans_HF[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
			# image = (image * 255).astype( np.uint8 )
			# plt.imshow( image )
			# plt.axis( 'off' )
			# # 保存图像到文件
			# plt.savefig( f'HF_enh_{i}.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		
		# HF_d + LF_l
		reconDL_pyrs = trans_pyrs.copy()
		reconDL_pyrs[ 0 ] = trans_pyrs_l[ 0 ]
		
		# HF_l + LF_d
		reconLD_pyrs = trans_pyrs_l.copy()
		reconLD_pyrs[ 0 ] = trans_pyrs[ 0 ]
		
		# HF_d + LF_d = trans_pyrs
		
		HF_d_LF_d = self.lap_pyramid.pyramid_recons( trans_pyrs )
		HF_l_LF_l = self.lap_pyramid.pyramid_recons( trans_pyrs_l )
		
		HF_d_LF_l = self.lap_pyramid.pyramid_recons( reconDL_pyrs )
		HF_l_LF_d = self.lap_pyramid.pyramid_recons( reconLD_pyrs )
		
		pyrs_d2 = self.lap_pyramid.pyramid_decom( img = HF_d_LF_l )
		pyrs_l2 = self.lap_pyramid.pyramid_decom( img = HF_l_LF_d )
		
		loss_equal = 0
		for i in range( self.num_high ) :
			loss_equal += F.mse_loss( trans_pyrs[ i+1 ],trans_pyrs_l[ i+1 ] )*0.1
		# HFe+LF_ori == img_ori
		loss_resume = F.mse_loss(HF_d_LF_d , x_dark.detach() )*1. +(1. - ssim(HF_d_LF_d, x_dark.detach()))
		loss_resume += F.mse_loss(HF_l_LF_l , x_light.detach() )*1.+(1. - ssim(HF_l_LF_l, x_light.detach()))
		
		loss_smooth = 0
		for i in range( self.num_high ) :
			loss_smooth += smooth(trans_pyrs[0],trans_pyrs[i+1],i+1)*cfg.WEIGHT.SMOOTH
			loss_smooth += smooth(trans_pyrs_l[0],trans_pyrs_l[i+1],i+1)*cfg.WEIGHT.SMOOTH
		
			losses_rc = (F.mse_loss( pyrs_d2[self.num_high-1-i] , trans_pyrs[i+1].detach() ) + F.mse_loss( pyrs_l2[self.num_high-1-i]  , trans_pyrs_l[i+1].detach() )) * cfg.WEIGHT.RC
		
		loss_enhance=loss_equal+loss_resume+loss_HFenh+losses_rc+loss_smooth
		
		# print( '最终的DL' )
		# image = np.transpose( HF_d_LF_l[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )  # 保存图像到文件
		# plt.savefig( f'DL.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )

		# print( '最终的LD' )
		# image = np.transpose( HF_l_LF_d[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )  # 保存图像到文件
		# plt.savefig( f'LD.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )

		# print( '最终的DD' )
		# image = np.transpose( HF_d_LF_d[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
		# image = (image * 255).astype( np.uint8 )
		# plt.imshow( image )
		# plt.axis( 'off' )# 保存图像到文件
		# plt.savefig( f'DD.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		# exit()
		
		return loss_enhance


class VGG16( nn.Module ) :
	def __init__( self , phase , base , extras , fem , head1 , head2 , num_classes ) :
		super().__init__()
		self.phase = phase
		self.num_classes = num_classes
		self.vgg = nn.ModuleList( base )  # 3个vgg16
		
		self.L2Normof1 = L2Norm( 256 , 10 )
		self.L2Normof2 = L2Norm( 512 , 8 )
		self.L2Normof3 = L2Norm( 512 , 5 )
		
		self.extras = nn.ModuleList( extras )
		self.fpn_topdown = nn.ModuleList( fem[ 0 ] )
		self.fpn_latlayer = nn.ModuleList( fem[ 1 ] )
		
		self.fpn_fem = nn.ModuleList( fem[ 2 ] )
		
		self.L2Normef1 = L2Norm( 256 , 10 )
		self.L2Normef2 = L2Norm( 512 , 8 )
		self.L2Normef3 = L2Norm( 512 , 5 )
		
		self.loc_pal1 = nn.ModuleList( head1[ 0 ] )  # nn.ModuleList是一种存储子模块的工具
		self.conf_pal1 = nn.ModuleList( head1[ 1 ] )
		
		self.loc_pal2 = nn.ModuleList( head2[ 0 ] )
		self.conf_pal2 = nn.ModuleList( head2[ 1 ] )
		
		self.KL = DistillKL( T = 1.0 )
		# self.mseloss=EnhanceLoss()
		
		if self.phase == 'test' :
			self.softmax = nn.Softmax( dim = -1 )
			self.detect = Detect( cfg )
	
	def forward( self , x , f_DD , f_LL , f_DL , f_LD ) :
		size = x.size()[ 2 : ]
		pal1_sources = list()
		loc_pal1 = list()
		conf_pal1 = list()
		loc_pal2 = list()
		conf_pal2 = list()
		
		# loss_enhloss = self.mseloss( f_DD , f_LL , f_DL , f_LD)
		
		for k in range( 16 ) :  # vgg13: 14 vgg16: 16
			x = self.vgg[ k ]( x )
		
		# 这里应该获得1,64,640,640的张量
		for i in range( 5 ) :
			f_DD = self.vgg[ i ]( f_DD )
		for i in range( 5 ) :
			f_DL = self.vgg[ i ]( f_DL )  # HF_d_LF_l
		for i in range( 5 ) :
			f_LL = self.vgg[ i ]( f_LL )  # HF_l_LF_l
		for i in range( 5 ) :
			f_LD = self.vgg[ i ]( f_LD )
		# f_LLe = self.vgg[ i ]( f_LLe )
		
		# 这里要控制LF相同，HF不同，使暗图高频增强有效果
		_f_DD = f_DD.flatten( start_dim = 2 ).mean( dim = -1 )
		_f_DL = f_DL.flatten( start_dim = 2 ).mean( dim = -1 )
		_f_LL = f_LL.flatten( start_dim = 2 ).mean( dim = -1 )
		_f_LD = f_LD.flatten( start_dim = 2 ).mean( dim = -1 )
		# _f_LLe = f_LLe.flatten( start_dim = 2 ).mean( dim = -1 )
		
		loss_mutual = cfg.WEIGHT.MC * (self.KL( _f_DD , _f_LL ) + self.KL( _f_LL , _f_DD )  # 高频优化有效
		                               # + self.KL( _f_LD , _f_LL ) + self.KL( _f_LL , _f_LD )  # 低频增强有效
		                               # + self.KL( _f_DL , _f_DD ) + self.KL( _f_DD , _f_DL )  # 正常亮度不会提高亮度
		                               )
		
		of1 = x  # 16个vgg后的输出
		s = self.L2Normof1( of1 )
		pal1_sources.append( s )
		# apply vgg up to fc7
		for k in range( 16 , 23 ) :  # vgg13: 14,19 vgg16: 16,23
			x = self.vgg[ k ]( x )
		of2 = x
		s = self.L2Normof2( of2 )
		pal1_sources.append( s )
		
		for k in range( 23 , 30 ) :  # vgg13: 19,24 vgg16: 23,30
			x = self.vgg[ k ]( x )
		of3 = x
		s = self.L2Normof3( of3 )
		pal1_sources.append( s )
		
		for k in range( 30 , len( self.vgg ) ) :  # vgg13: 24 vgg16: 30
			x = self.vgg[ k ]( x )
		of4 = x
		pal1_sources.append( of4 )
		
		for k in range( 2 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of5 = x
		pal1_sources.append( of5 )
		for k in range( 2 , 4 ) :
			x = F.relu( self.extras[ k ]( x ) , inplace = True )
		of6 = x
		pal1_sources.append( of6 )
		
		conv7 = F.relu( self.fpn_topdown[ 0 ]( of6 ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 1 ]( conv7 ) , inplace = True )
		conv6 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 0 ]( of5 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 2 ]( conv6 ) , inplace = True )
		convfc7_2 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 1 ]( of4 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 3 ]( convfc7_2 ) , inplace = True )
		conv5 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 2 ]( of3 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 4 ]( conv5 ) , inplace = True )
		conv4 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 3 ]( of2 ) ) , inplace = True )
		
		x = F.relu( self.fpn_topdown[ 5 ]( conv4 ) , inplace = True )
		conv3 = F.relu( self._upsample_prod( x , self.fpn_latlayer[ 4 ]( of1 ) ) , inplace = True )
		
		ef1 = self.fpn_fem[ 0 ]( conv3 )
		ef1 = self.L2Normef1( ef1 )
		ef2 = self.fpn_fem[ 1 ]( conv4 )
		ef2 = self.L2Normef2( ef2 )
		ef3 = self.fpn_fem[ 2 ]( conv5 )
		ef3 = self.L2Normef3( ef3 )
		ef4 = self.fpn_fem[ 3 ]( convfc7_2 )
		ef5 = self.fpn_fem[ 4 ]( conv6 )
		ef6 = self.fpn_fem[ 5 ]( conv7 )
		
		pal2_sources = (ef1 , ef2 , ef3 , ef4 , ef5 , ef6)
		for (x , l , c) in zip( pal1_sources , self.loc_pal1 , self.conf_pal1 ) :
			loc_pal1.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal1.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		for (x , l , c) in zip( pal2_sources , self.loc_pal2 , self.conf_pal2 ) :
			loc_pal2.append( l( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			conf_pal2.append( c( x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
		
		features_maps = [ ]
		for i in range( len( loc_pal1 ) ) :
			feat = [ ]
			feat += [ loc_pal1[ i ].size( 1 ) , loc_pal1[ i ].size( 2 ) ]
			features_maps += [ feat ]
		
		loc_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal1 ] , 1 )
		conf_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal1 ] , 1 )
		
		loc_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in loc_pal2 ] , 1 )
		conf_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 ) for o in conf_pal2 ] , 1 )
		
		priorbox = PriorBox( size , features_maps , cfg , pal = 1 )
		self.priors_pal1 = priorbox.forward().requires_grad_( False )
		priorbox = PriorBox( size , features_maps , cfg , pal = 2 )
		self.priors_pal2 = priorbox.forward().requires_grad_( False )
		
		if self.phase == 'test' :
			output = self.detect.forward( loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) , self.softmax(
					conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) ) ,  # conf preds
			                              self.priors_pal2.type( type( x.data ) ) )
		
		else :
			output = (loc_pal1.view( loc_pal1.size( 0 ) , -1 , 4 ) ,
			          conf_pal1.view( conf_pal1.size( 0 ) , -1 , self.num_classes ) , self.priors_pal1 ,
			          loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
			          conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) , self.priors_pal2)
		
		# out,loss_mutual = self.backbone( x1,Light_dark=x1,Dark_light=Dark_light )
		
		return output , loss_mutual
	
	def _upsample_prod( self , x , y ) :
		_ , _ , H , W = y.size()
		return F.interpolate( x , size = (H , W) , mode = 'bilinear' ) * y


class DSFD( nn.Module ) :
	def __init__( self , phase , base , extras , fem , head1 , head2 , num_classes ) :
		super().__init__()
		# enhancement
		self.enhancement = DENet()
		# backbone
		# self.backbone = VGG16( phase , base , extras , fem , head1 , head2 , num_classes )
	
	def test_forward( self , x_dark , x_light ) :
		with torch.no_grad() :
			loss_enhance = self.enhancement( x_dark = x_dark , x_light = x_light )  # 返回值是归一的
			
			return loss_enhance
	
	def forward( self , x_dark , x_light ) :
		
		loss_enhance = self.enhancement( x_dark = x_dark ,x_light = x_light )  # 返回值是归一的
		
		return loss_enhance
	
	def load_weights( self , base_file ) :
		other , ext = os.path.splitext( base_file )
		if ext == '.pkl' or '.pth' :
			print( 'Loading weights into state dict...' )
			mdata = torch.load( base_file , map_location = lambda storage , loc : storage )
			
			epoch = 0
			self.load_state_dict( mdata )
			print( 'Finished!' )
		else :
			print( 'Sorry only .pth and .pkl files supported.' )
		del other , ext , mdata
		torch.cuda.empty_cache()
		return epoch
	
	def xavier( self , param ) :
		init.xavier_uniform_( param )
	
	def weights_init( self , m ) :
		if isinstance( m , nn.Conv2d ) :
			self.xavier( m.weight.data )
			if 'bias' in m.state_dict().keys() :
				m.bias.data.zero_()
		
		if isinstance( m , nn.ConvTranspose2d ) :
			self.xavier( m.weight.data )
			if 'bias' in m.state_dict().keys() :
				m.bias.data.zero_()
		
		if isinstance( m , nn.BatchNorm2d ) :
			m.weight.data[ ... ] = 1
			m.bias.data.zero_()
	
	def _upsample_prod( self , x , y ) :
		_ , _ , H , W = y.size()
		return F.interpolate( x , size = (H , W) , mode = 'bilinear' ) * y


vgg_cfg = [ 64 , 64 , 'M' , 128 , 128 , 'M' , 256 , 256 , 256 , 'C' , 512 , 512 , 512 , 'M' , 512 , 512 , 512 , 'M' ]
# vgg_cfg = [ 64 , 64 , 'M' , 128 , 128 , 'M' , 256 , 256 , 'C' , 512 , 512 , 'M' , 512 , 512 , 'M' ]
extras_cfg = [ 256 , 'S' , 512 , 128 , 'S' , 256 ]

fem_cfg = [ 256 , 512 , 512 , 1024 , 512 , 256 ]


def fem_module( cfg ) :
	topdown_layers = [ ]
	lat_layers = [ ]
	fem_layers = [ ]
	
	topdown_layers += [ nn.Conv2d( cfg[ -1 ] , cfg[ -1 ] , kernel_size = 1 , stride = 1 , padding = 0 ) ]
	for k , v in enumerate( cfg ) :
		fem_layers += [ FEM( v ) ]
		cur_channel = cfg[ len( cfg ) - 1 - k ]
		if len( cfg ) - 1 - k > 0 :
			last_channel = cfg[ len( cfg ) - 2 - k ]
			topdown_layers += [ nn.Conv2d( cur_channel , last_channel , kernel_size = 1 , stride = 1 , padding = 0 ) ]
			lat_layers += [ nn.Conv2d( last_channel , last_channel , kernel_size = 1 , stride = 1 , padding = 0 ) ]
	return (topdown_layers , lat_layers , fem_layers)


def vgg( cfg , i , batch_norm = False ) :
	layers = [ ]
	in_channels = i
	for v in cfg :
		if v == 'M' :
			layers += [ nn.MaxPool2d( kernel_size = 2 , stride = 2 ) ]
		elif v == 'C' :
			layers += [ nn.MaxPool2d( kernel_size = 2 , stride = 2 , ceil_mode = True ) ]
		else :
			conv2d = nn.Conv2d( in_channels , v , kernel_size = 3 , padding = 1 )
			if batch_norm :
				layers += [ conv2d , nn.BatchNorm2d( v ) , nn.ReLU( inplace = True ) ]
			else :
				layers += [ conv2d , nn.ReLU( inplace = True ) ]
			in_channels = v
	conv6 = nn.Conv2d( 512 , 1024 , kernel_size = 3 , padding = 3 , dilation = 3 )
	conv7 = nn.Conv2d( 1024 , 1024 , kernel_size = 1 )
	layers += [ conv6 , nn.ReLU( inplace = True ) , conv7 , nn.ReLU( inplace = True ) ]
	return layers


def add_extras( cfg , i , batch_norm = False ) :
	# Extra layers added to VGG for feature scaling
	layers = [ ]
	in_channels = i
	flag = False
	for k , v in enumerate( cfg ) :
		if in_channels != 'S' :
			if v == 'S' :
				layers += [ nn.Conv2d( in_channels , cfg[ k + 1 ] , kernel_size = (1 , 3)[ flag ] , stride = 2 ,
				                       padding = 1 ) ]
			else :
				layers += [ nn.Conv2d( in_channels , v , kernel_size = (1 , 3)[ flag ] ) ]
			flag = not flag
		in_channels = v
	return layers


def multibox( vgg , extra_layers , num_classes ) :
	loc_layers = [ ]
	conf_layers = [ ]
	vgg_source = [ 14 , 21 , 28 , -2 ]  # vgg13: [ 12 , 17 , 22 , -2 ] vgg16: [14, 21, 28, -2]
	
	for k , v in enumerate( vgg_source ) :
		# print(v)
		loc_layers += [ nn.Conv2d( vgg[ v ].out_channels , 4 , kernel_size = 3 , padding = 1 ) ]
		conf_layers += [ nn.Conv2d( vgg[ v ].out_channels , num_classes , kernel_size = 3 , padding = 1 ) ]
	for k , v in enumerate( extra_layers[ 1 : :2 ] , 2 ) :
		loc_layers += [ nn.Conv2d( v.out_channels , 4 , kernel_size = 3 , padding = 1 ) ]
		conf_layers += [ nn.Conv2d( v.out_channels , num_classes , kernel_size = 3 , padding = 1 ) ]
	return (loc_layers , conf_layers)


def build_net_dark( phase , num_classes = 2 ) :
	base = vgg( vgg_cfg , 3 )
	extras = add_extras( extras_cfg , 1024 )
	head1 = multibox( base , extras , num_classes )
	head2 = multibox( base , extras , num_classes )
	fem = fem_module( fem_cfg )
	return DSFD( phase , base , extras , fem , head1 , head2 , num_classes )


class DistillKL( nn.Module ) :
	"""KL divergence for distillation"""
	
	# 知识蒸馏模块，处理KL散度
	def __init__( self , T ) :
		super( DistillKL , self ).__init__()
		self.T = T
	
	def forward( self , y_s , y_t ) :
		# y_s学生模型的输出，y_t 教师模型的输出
		p_s = F.log_softmax( y_s / self.T , dim = 1 )  # 对数概率分布
		p_t = F.softmax( y_t / self.T , dim = 1 )  # 概率分布
		# 计算KL散度
		# size_average不使用平均损失，而是返回总损失，(self.T ** 2)补偿温度缩放，/ y_s.shape[0]计算平均损失
		loss = F.kl_div( p_s , p_t , size_average = False ) * (self.T ** 2) / y_s.shape[ 0 ]
		return loss
