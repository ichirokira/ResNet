# ResNet

ResNet Model is based on a Shortcut Connection with the purpose to help DeepLearning model to learn deeper without vanishing gradient

In ResNet Architecture, specifically in a Residual Module:
  -  Consist of 3 groups, each group contains: BN + Act(Relu) + Conv2D(K)
     and the group1 and group2 the channel_dim in Conv2D layer is equals 1/4 channel_dim in group3
  -  from original input, it goes through a BatchNormal layer as an added level of nomalization. Also, removing mean normalization step

NOTE: Since the effect of identicals mapping, when  training ResNet we should apply high learning rate, may be = 0.1
I also introduce a method to linear weight decay is poly-decay (new_alpja = init_alpha * (epoch/numEpoch)*power)


The ResNet model is support for Cifar and TinyImageNet Challenge
