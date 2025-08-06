import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
# **********************************V1:ATTENTION*******************************************************##
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, inplanes):
        super(PAM_Module, self).__init__()
        self.inplanes = inplanes
        self.query_conv = nn.Conv2d(in_channels=inplanes, out_channels=inplanes//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=inplanes, out_channels=inplanes//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( N X C X H X W)
            returns :
                out : attention value + input feature
                attention: N X (HxW) X (HxW)
        """
        N, C, H, W = x.size()
        proj_query = self.query_conv(x).view(N, -1, W*H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(N, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(N, -1, W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        del proj_query, proj_key, energy, attention
        torch.cuda.empty_cache()

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, inplanes):
        super(CAM_Module, self).__init__()
        self.inplanes = inplanes
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        N, C, H, W = x.size()
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(N, C, -1)

        out = torch.bmm(attention, proj_value)
        del proj_query, proj_key, energy, attention
        torch.cuda.empty_cache()
        out = out.view(N, C, H, W)

        out = self.gamma*out + x
        return out
    
class Fusion_Module(nn.Module):
    def __init__(self, inplanes):
        super(Fusion_Module,self).__init__()

        self.inplanes = inplanes
        self.pam_attention = PAM_Module(inplanes)
        self.cam_attention = CAM_Module(inplanes)

    def forward(self,x):
       fusion = self.pam_attention(x) + self.cam_attention(x)
       return fusion
#**********************************V1:ATTENTION****************************************************##

##**********************************V2:Scconv*******************************************************##
# 自定义 GroupBatchnorm2d 类，实现分组批量归一化
class GroupBatchnorm2d(nn.Module):
  def __init(self, c_num:int,
            group_num:int = 16,
            eps:float = 1e-10):
    super(GroupBatchnorm2d,self).__init__()# 调用父类构造函数
    assert c_num >= group_num # 断言 c_num 大于等于 group_num
    self.group_num = group_num # 设置分组数量
    self.weight = nn.Parameter(torch.randn(c_num, 1, 1))# 创建可训练参数 gamma
    self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))# 创建可训练参数 beta
    self.eps = eps ## 设置小的常数 eps 用于稳定计算

  def forward(self,x):
    N, C, H, W  = x.size() # 获取输入张量的尺寸
    x  = x.view(N, self.group_num, -1) # 将输入张量重新排列为指定的形状
    mean = x.mean(dim = 2, keepdim = True) # 计算每个组的均值
    std = x.std (dim = 2, keepdim = True) # 计算每个组的标准差
    x = (x - mean) / (std+self.eps) # 应用批量归一化
    x  = x.view(N, C, H, W) # 恢复原始形状
    return x * self.weight + self.bias # 返回归一化后的张量
  
class SRU(nn.Module):
  def __init__(self,
                oup_channels:int,           # 输出通道数
                group_num:int = 16,         # 分组数，默认为16
                gate_treshold:float = 0.5,  # 门控阈值，默认为0.5
                torch_gn:bool = True):      # 是否使用PyTorch内置的GroupNorm，默认为False
    super().__init__()  # 调用父类构造函数
    # 初始化 GroupNorm 层或自定义 GroupBatchnorm2d 层    
    self.gn = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
    self.gate_treshold = gate_treshold # 设置门控阈值
    self.sigomid = nn.Sigmoid() # 创建 sigmoid 激活函数

  def forward(self,x):
    gn_x = self.gn(x) # 应用分组批量归一化
    w_gamma = self.gn.weight/sum(self.gn.weight) # 计算 gamma 权重
    w_gamma  = w_gamma.view(1,-1,1,1)
    reweigts  = self.sigomid( gn_x * w_gamma )# 计算重要性权重
    # Gate
    w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
    w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
    x_1 = w1 * x
    x_2  = w2 * x
    y  = self.reconstruct(x_1,x_2)# 重构特征

    return y

  def reconstruct(self,x_1,x_2):
    x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
    x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
    return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)

class CRU(nn.Module):
  def __init__(self,op_channel:int,alpha:float = 1/2,squeeze_radio:int = 2 ,group_size:int = 2,group_kernel_size:int = 3):##squeeze_radio压缩比:2 alpha分割比
    super().__init__()
    self.up_channel = up_channel = int(alpha*op_channel)# 计算上层通道数
    self.low_channel = low_channel = op_channel-up_channel# 计算下层通道数
    self.squeeze1 = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False) #上层压缩后
    self.squeeze2 = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
    #up 上层特征转换
    self.GWC = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)#分组卷积
    self.PWC1 = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)#点卷积
    #low
    self.PWC2 = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
    self.advavg = nn.AdaptiveAvgPool2d(1)

  def forward(self,x):
    # Split
    up,low = torch.split(x,[self.up_channel,self.low_channel],dim=1)
    up,low = self.squeeze1(up),self.squeeze2(low)
    # Transform
    Y1 = self.GWC(up) + self.PWC1(up)
    Y2 = torch.cat( [self.PWC2(low), low], dim= 1 )
    # Fuse
    out = torch.cat( [Y1,Y2], dim= 1 )
    out = F.softmax( self.advavg(out), dim=1 ) * out
    out1,out2 = torch.split(out,out.size(1)//2,dim=1)
    return out1+out2

# 自定义 ScConv模型
class ScConv(nn.Module):
    def __init__(self, op_channel:int, group_num:int = 16, gate_treshold:float = 0.5, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()  # 调用父类构造函数
 
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)  # 创建 SRU 层
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size, group_kernel_size=group_kernel_size)  # 创建 CRU 层
 
    def forward(self, x):
        x = self.SRU(x)  # 应用 SRU 层
        x = self.CRU(x)  # 应用 CRU 层
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class Final_Model(nn.Module):

    def __init__(self, backbone_net, semantic_head):
        super(Final_Model, self).__init__()
        self.backend = backbone_net
        self.semantic_head = semantic_head

    def forward(self, x):
        middle_feature_maps = self.backend(x)

        semantic_output = self.semantic_head(middle_feature_maps)

        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_34(nn.Module):
    def __init__(self, nclasses, aux, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64):
        super(ResNet_34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128
        #
        self.scconv = ScConv(128)

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)
    
    def _make_layer2(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        #
        layers.append((Fusion_Module(planes)))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.scconv(x)

        x_1 = self.layer1(x)  # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            res_2 = self.aux_head1(res_2)
            res_2 = F.softmax(res_2, dim=1)

            res_3 = self.aux_head2(res_3)
            res_3 = F.softmax(res_3, dim=1)

            res_4 = self.aux_head3(res_4)
            res_4 = F.softmax(res_4, dim=1)

#             res_2 = self.aux_head1(x_2)
#             res_2 = F.softmax(x_2, dim=1)

#             res_3 = self.aux_head2(x_3)
#             res_3 = F.softmax(x_3, dim=1)

#             res_4 = self.aux_head3(x_4)
#             res_4 = F.softmax(x_4, dim=1)

        if self.aux:
            return [out, res_2, res_3, res_4]
        else:
            return out





if __name__ == "__main__":
    import time
    model = ResNet_34(20).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    for i in range(20):
        inputs = torch.randn(1, 5, 64, 2048).cuda()
        model.eval()
        with torch.no_grad():
          start_time = time.time()
          outputs = model(inputs)
        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        fwt = time.time() - start_time
        time_train.append(fwt)
        print ("Forward time per img: %.3f (Mean: %.3f)" % (
          fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)




