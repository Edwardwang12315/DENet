from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from data.config import cfg


def gradient( input_tensor , direction ) :
	smooth_kernel_x = torch.FloatTensor( [ [ 0 , 0 ] , [ -1 , 1 ] ] ).view( (1 , 1 , 2 , 2) ).cuda()
	smooth_kernel_y = torch.transpose( smooth_kernel_x , 2 , 3 )
	
	if direction == "x" :
		kernel = smooth_kernel_x
	elif direction == "y" :
		kernel = smooth_kernel_y
	grad_out = torch.abs( F.conv2d( input_tensor , kernel ,
	                                stride = 1 , padding = 1 ) )
	return grad_out


def ave_gradient( input_tensor , direction ) :
	return F.avg_pool2d( gradient( input_tensor , direction ) ,
	                     kernel_size = 3 , stride = 1 , padding = 1 )


def smooth( input_LF , input_HF,sample ) :
	input_HF = 0.299 * input_HF[ : , 0 , : :2**sample  , : :2**sample ] + 0.587 * input_HF[ : , 1 , : :2**sample  , : :2**sample  ] + 0.114 * input_HF[ : , 2 , : :2**sample  , : :2**sample ]
	input_HF = torch.unsqueeze( input_HF , dim = 1 )
	input_LF = 0.299 * input_LF[ : , 0 , : , : ] + 0.587 * input_LF[ : , 1 , : , : ] + 0.114 * input_LF[ : , 2 , : , : ]
	input_LF = torch.unsqueeze( input_LF , dim = 1 )
	
	return torch.mean( gradient( input_LF , "x" ) * torch.exp( -10 * ave_gradient( input_HF , "x" ) ) +
	                   gradient( input_LF , "y" ) * torch.exp( -10 * ave_gradient( input_HF , "y" ) ) )

class EnhanceLoss( nn.Module ) :
	def __init__( self ) :
		super().__init__()
	
	def forward( self , pyrs_enh , pyrs_tar,DD,LL) :
		LF_enh , HF3_enh , HF2_enh , HF1_enh = pyrs_enh
		HF1_tar , HF2_tar , HF3_tar , LF_tar = pyrs_tar
		
		# 约束enh和tar的差异
		losses_LF = (F.mse_loss(LF_enh, LF_tar.detach()))
		
		losses_HFs= F.l1_loss( HF1_enh , HF1_tar.detach() )+(1. - ssim( HF1_enh , HF1_tar.detach() )
		            +F.l1_loss( HF2_enh , HF2_tar.detach() )+ (1. - ssim( HF2_enh , HF2_tar.detach() ))
		            + F.l1_loss( HF3_enh , HF3_tar.detach() )+(1. - ssim( HF3_enh , HF3_tar.detach() ))) * cfg.WEIGHT.EQUAL_R
		
		loss = (F.mse_loss( DD , LL.detach() )+ (1. - ssim( DD , LL.detach() )))* cfg.WEIGHT.EQUAL_R
		
		enhance_loss = (losses_LF + losses_HFs + loss) / 3
		
		# print( "losses_LF:", losses_LF)
		# print( "losses_HFs:", losses_HFs)
		# print( "loss:", loss)
		return enhance_loss
