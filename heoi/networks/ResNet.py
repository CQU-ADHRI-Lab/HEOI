import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # if kernel = 3, stride = 1, pad = 1, then output == input
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    # there are two residual block types, when the layers are less than 50,
    # we use basicblock type
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # there are two residual block types, when the layers are larger than 50,
    # we use bottleneck type
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)  #
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet18(pretrained=True, classes=10, channel=3, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], nb_classes=classes,
                   channel=channel, **kwargs)
    # layers = 2*(2+2+2+2) + 2(输入输出)
    if pretrained:
        pretrain_dict = model_zoo.load_url(
            model_urls['resnet18'])  # modify pretrain code
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model

def resnet18_AllLayers(pretrained=True, classes=10, channel=3, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_AllLayers(BasicBlock, [2, 2, 2, 2], nb_classes=classes,
                   channel=channel, **kwargs)
    # layers = 2*(2+2+2+2) + 2(输入输出)
    if pretrained:
        pretrain_dict = model_zoo.load_url(
            model_urls['resnet18'])  # modify pretrain code
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=True, classes=10, channel=3, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], nb_classes=classes,
                   channel=channel, **kwargs)
    # 34 = 2*（3+4+6+3）+2，　每个ｂｌｏｃｋ是２层，　
    if pretrained:
        pretrain_dict = model_zoo.load_url(
            model_urls['resnet34'])  # modify pretrain code
        model_dict = model.state_dict()
        # print('hhhhh', model_dict)
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=True, classes=10, channel=3, **kwargs):
    model = ResNetV2(Bottleneck, [3, 4, 6, 3], nb_classes=classes,
                   channel=channel, **kwargs)
    # 50 = 3*(3+4+6+3) + 2 , the bottleneck block has 3 layers
    if pretrained:
        pretrain_dict = model_zoo.load_url(
            model_urls['resnet50'])  # modify pretrain code
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=True, classes=2, channel=5, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], nb_classes=classes,
                   channel=channel, **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(
            model_urls['resnet101'])  # modify pretrain code
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)

    return model


def resnet152(pretrained=True, classes=10, channel=3, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], nb_classes=classes,
                   channel=channel, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        pretrain_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict, channel)
        model.load_state_dict(model_dict)
    return model


class ResNet(nn.Module):  # nn.Module is the inherit class
    def __init__(self, block, layers, nb_classes=2, channel=5):
        # block is the Basic block or Bottleneck block，　
        self.inplanes = 64
        super(ResNet, self).__init__()
        # find the father of ResNet
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2,
                                      padding=3,
                                      bias=False)
        # parameter1 is input channel, parameter2 is output channels,　
        # 第一层卷积层的卷积核种类是20， 输出是64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        # Linear is fully connected layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果参数１的类型是参数２的类型相同，返回True
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def fm_extract(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # block is the basic block or bottleneck block
        # planes is the output channel of block，该参数指定的是卷积核的个数，
        # ［3，4，6，3］把网络分为４个大的ｂｌｏｃｋ，每个大的ｂｌｏｃｋ中的
        # 小的blocks的卷积核个数是固定的，分别是64，　128，　256，　512
        # blocks is the number of block
        downsample = None
        # block.expansion == 1 or 4,  1 is for basic, 4 is for bottleneck
        # 第一层的输出固定为64，　又因为第一个大的ｂｌｏｃｋ的
        # stride != 1 是指从第２个ｌａｙｅｒ开始，
        # self.inplanes != planes * block.expansion
        #      planes == 64，　128，　256，　512
        # self.inplane== 64,  256,   512,   1024
        out_planes = planes * block.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            # print('expansion', block.expansion)
            # print('self.inplane', self.inplanes)

        # _make_layer方法中比较重要的两行代码是：
        # 1、layers.append(block(self.inplanes, planes, stride, downsample))，
        # 该部分是将每个blocks的第一个residual结构保存在layers列表中。
        # 2、 for i in range(1, block(self.inplanes, planes))，
        # 该部分是将每个blocks的剩下residual结构保存在layers列表中，
        # 这样就完成了一个blocks的构造。
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample))  # (in_out),(out_out),(out_4*out)
        # 有ｄｏｗｎｓａｍｐｌｅ，
        self.inplanes = planes * block.expansion  # self.inplans 在上一次（上一行）layers.append后就变大了，上一层planes通常小，比如128-256  256-256  
        # 没有ｄｏｗｎｓａｍｐｌｅ，但是　inplanes = 4*planes,
        # 所以ｄｏｗｎｓａｍｐｌｅ要保证输出分别是64*4，　128*4，　256*4，　512*4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class ResNet_AllLayers(nn.Module):  # nn.Module is the inherit class
    def __init__(self, block, layers, nb_classes=2, channel=5):
        # block is the Basic block or Bottleneck block，　
        self.inplanes = 64
        super(ResNet_AllLayers, self).__init__()
        # find the father of ResNet
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2,
                                      padding=3,
                                      bias=False)
        # parameter1 is input channel, parameter2 is output channels,　
        # 第一层卷积层的卷积核种类是20， 输出是64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        # Linear is fully connected layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果参数１的类型是参数２的类型相同，返回True
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.x_list = []
    def fm_extract(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # self.x_list.append[x]
        x = self.layer1(x)
        d1 = x
        # self.x_list.append[x]
        x = self.layer2(x)
        d2 = x
        # self.x_list.append[x]
        x = self.layer3(x)
        d3 = x
        # self.x_list.append[x]
        x = self.layer4(x)
        d4 = x
        # self.x_list.append[x]

        return [d1, d2, d3, d4]

    def _make_layer(self, block, planes, blocks, stride=1):
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # block is the basic block or bottleneck block
        # planes is the output channel of block，该参数指定的是卷积核的个数，
        # ［3，4，6，3］把网络分为４个大的ｂｌｏｃｋ，每个大的ｂｌｏｃｋ中的
        # 小的blocks的卷积核个数是固定的，分别是64，　128，　256，　512
        # blocks is the number of block
        downsample = None
        # block.expansion == 1 or 4,  1 is for basic, 4 is for bottleneck
        # 第一层的输出固定为64，　又因为第一个大的ｂｌｏｃｋ的
        # stride != 1 是指从第２个ｌａｙｅｒ开始，
        # self.inplanes != planes * block.expansion
        #      planes == 64，　128，　256，　512
        # self.inplane== 64,  256,   512,   1024
        out_planes = planes * block.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            # print('expansion', block.expansion)
            # print('self.inplane', self.inplanes)

        # _make_layer方法中比较重要的两行代码是：
        # 1、layers.append(block(self.inplanes, planes, stride, downsample))，
        # 该部分是将每个blocks的第一个residual结构保存在layers列表中。
        # 2、 for i in range(1, block(self.inplanes, planes))，
        # 该部分是将每个blocks的剩下residual结构保存在layers列表中，
        # 这样就完成了一个blocks的构造。
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample))  # (in_out),(out_out),(out_4*out)
        # 有ｄｏｗｎｓａｍｐｌｅ，
        self.inplanes = planes * block.expansion
        # 没有ｄｏｗｎｓａｍｐｌｅ，但是　inplanes = 4*planes,
        # 所以ｄｏｗｎｓａｍｐｌｅ要保证输出分别是64*4，　128*4，　256*4，　512*4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
class ResNetV2(nn.Module):  # nn.Module is the inherit class
    def __init__(self, block, layers, nb_classes=2, channel=5):
        # block is the Basic block or Bottleneck block，　
        self.inplanes = 64
        super(ResNetV2, self).__init__()
        # find the father of ResNet
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2,
                                      padding=3,
                                      bias=False)
        # parameter1 is input channel, parameter2 is output channels,　
        # 第一层卷积层的卷积核种类是20， 输出是64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc_custom = nn.Linear(512 * block.expansion, 512)
        self.conv_custom = nn.Conv2d(512 * block.expansion, 512,
                                     kernel_size=1, stride=1, bias=False)
        # Linear is fully connected layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果参数１的类型是参数２的类型相同，返回True
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def fm_extract(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_custom(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # block is the basic block or bottleneck block
        # planes is the output channel of block，该参数指定的是卷积核的个数，
        # ［3，4，6，3］把网络分为４个大的ｂｌｏｃｋ，每个大的ｂｌｏｃｋ中的
        # 小的blocks的卷积核个数是固定的，分别是64，　128，　256，　512
        # blocks is the number of block
        downsample = None
        # block.expansion == 1 or 4,  1 is for basic, 4 is for bottleneck
        # 第一层的输出固定为64，　又因为第一个大的ｂｌｏｃｋ的
        # stride != 1 是指从第２个ｌａｙｅｒ开始，
        # self.inplanes != planes * block.expansion
        #      planes == 64，　128，　256，　512
        # self.inplane== 64,  256,   512,   1024
        out_planes = planes * block.expansion
        # print('expansion', block.expansion)
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            # print('self.inplane', self.inplanes)

        # _make_layer方法中比较重要的两行代码是：
        # 1、layers.append(block(self.inplanes, planes, stride, downsample))，
        # 该部分是将每个blocks的第一个residual结构保存在layers列表中。
        # 2、 for i in range(1, block(self.inplanes, planes))，
        # 该部分是将每个blocks的剩下residual结构保存在layers列表中，
        # 这样就完成了一个blocks的构造。
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample))  # (in_out),(out_out),(out_4*out)
        # 有ｄｏｗｎｓａｍｐｌｅ，
        self.inplanes = planes * block.expansion
        # 没有ｄｏｗｎｓａｍｐｌｅ，但是　inplanes = 4*planes,
        # 所以ｄｏｗｎｓａｍｐｌｅ要保证输出分别是64*4，　128*4，　256*4，　512*4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    # in order to utilize ImageNet pre-trained weight on our model,
    # we have to modify the weights of
    # the first convolution layer pre-trained with ImageNet from
    # (64, 3, 7, 7) to (64, 20, 7, 7).
    S = 0
    for i in range(3):
        S += conv1_weight[:, i, :, :]
    avg = S / 3.
    new_conv1_weight = torch.FloatTensor(64, channel, 7, 7)
    # print type(avg),type(new_conv1_weight)
    # print('AVERAGE',type(avg),avg.size(), type(S))
    for i in range(channel):
        new_conv1_weight[:, i, :, :] = avg.data
    return new_conv1_weight


def weight_transform(model_dict, pretrain_dict, channel):
    # model_dict = model.state_dict()
    # 返回一个字典，　['bias', 'weight']　
    # model_dict = weight_transform(model_dict, pretrain_dict, channel)
    # model.load_state_dict(model_dict)
    # print(type(model_dict))
    # print(model_dict.keys())
    # print(type(pretrain_dict))
    weight_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    # print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    # 因为输入数据的channel是３，　当输入２０个通道的optical　flow数据的时候，
    # 模型参数不统一，所以要调整
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3, channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)

    return model_dict


if __name__ == "__main__":
    res50 = resnet34(pretrained=True)
    a = torch.FloatTensor(2, 3, 7, 7)
    b = res50.fm_extract(a)

