import torch.nn as nn
import torchvision
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.nn.functional as F
from Model import viTransformer
from torch.nn.init import xavier_uniform_

class Attention(nn.Module):
    def __init__(self, feat_size, num_classes):
        super().__init__()
        self.feat_size = feat_size
        self.num_classes = num_classes

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(feat_size, num_classes)
        xavier_uniform_(self.U.weight)
        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(feat_size, num_classes)
        xavier_uniform_(self.final.weight)


    def forward(self, x):

        # nonlinearity (tanh)
        x = torch.tanh(x)
        # apply attention
        alpha = torch.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        # imgage representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return y

class DenseNet121(nn.Module):
    def __init__(self, num_labels):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_features, num_labels)

    def forward(self, x):
        x = self.densenet121(x)

        return x


class MultiTaskDenseNet121(nn.Module):
    def __init__(self, num_labels_per_task):
        super(MultiTaskDenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_features = self.densenet121.classifier.in_features
        self.num_labels_per_task = num_labels_per_task
        self.fc = nn.ModuleDict(
            {task_name: nn.Linear(num_features, task_dim) for task_name, task_dim in self.num_labels_per_task.items()}
        )


    def forward(self, x):
        features=self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        x ={task: layer(out) for task, layer in self.fc.items()}
        return x


class ClassificationModel(nn.Module):
    def __init__(self, num_labels_per_task):
        super(ClassificationModel, self).__init__()
        self.num_labels = len(num_labels_per_task)
        # image encoder
        # self.dropout = nn.Dropout()
        self.visual_features = DenseNet121(self.num_labels)


    def forward(
        self,
        img=None,
        labels=None,
        n_crops=None,
        batch_size=None,
    ):

        logits= self.visual_features(img)
        if n_crops is not None:
            logits = logits.view(batch_size, n_crops, -1).mean(1)

        # logits = self.fc(img_feat_)
        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.float())
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class MultiTaskClassificationModel(nn.Module):
    def __init__(self, num_labels_per_task):
        super(MultiTaskClassificationModel, self).__init__()
        self.num_labels_per_task = num_labels_per_task
        # image encoder
        #self.dropout = nn.Dropout()
        self.visual_features = MultiTaskViTransfer(num_labels_per_task)

    def forward(
        self,
        img=None,
        labels=None,
        n_crops=None,
        batch_size=None,
        finger_index=None,
    ):

        logits = self.visual_features(img, finger_index=finger_index)
        if n_crops is not None:
            logits = {task:x.view(batch_size, n_crops, -1).mean(1)  for task,x in logits.items()}

        outputs = (logits,)
        losses = {}

        for idx, task in enumerate(self.num_labels_per_task.keys()):
            p=logits[task].view(-1, self.num_labels_per_task[task])
            gt=labels[:,idx].view(-1)
            losses[task] = nn.CrossEntropyLoss()(p,gt)

        outputs = (losses,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class ViTransferClassification(nn.Module):
    def __init__(self, num_labels_per_task):
        super(ViTransferClassification, self).__init__()
        self.num_labels = len(num_labels_per_task)
        # image encoder
        self.visual_features = viTransformer.vit_large_patch32_384(pretrained=True, num_classes=self.num_labels)

    def forward(
        self,
        img=None,
        labels=None,
        n_crops=None,
        batch_size=None,
        finger_index=None
    ):
        logits = self.visual_features(img, finger_index)
        if n_crops is not None:
            logits = logits.view(batch_size, n_crops, -1).mean(1)

        # logits = self.fc(img_feat_)
        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.float())
            outputs = (loss,) + outputs
        return outputs


class MultiTaskViTransfer(nn.Module):
    def __init__(self, num_labels_per_task):
        self.num_labels = len(num_labels_per_task)
        super(MultiTaskViTransfer, self).__init__()
        self.finger_index_embeddings = nn.Embedding(num_embeddings=20, embedding_dim=1024)

        self.visual_features = viTransformer.vit_large_patch32_384(pretrained=True, num_classes=self.num_labels)
        num_features = self.visual_features.num_features
        self.num_labels_per_task = num_labels_per_task
        self.fc = nn.ModuleDict(
            {task_name: nn.Linear(num_features, task_dim) for task_name, task_dim in self.num_labels_per_task.items()}
        )

    def forward(
        self,
        x,
        finger_index=None
    ):
        out = self.visual_features.forward_features(x)
        x = {task: layer(out) for task, layer in self.fc.items()}

        return x


class ViTransferClassificationLayers(nn.Module):
    def __init__(self, num_labels_per_task):
        super(ViTransferClassificationLayers, self).__init__()
        self.num_labels = len(num_labels_per_task)
        # image encoder
        self.visual_features = viTransformer.vit_large_patch32_384(pretrained=True, num_classes=self.num_labels)

        # dense + batchnorm + relu
        fc1=nn.Linear(self.visual_features.num_features, 512)
        self.fc1=nn.Sequential(fc1,nn.BatchNorm1d(512),nn.ReLU())
        fc2=nn.Linear(512, 256)
        self.fc2=nn.Sequential(fc2,nn.BatchNorm1d(256),nn.ReLU())
        self.fc3 = nn.Linear(256, self.num_labels)
        #self.fc4 = nn.Linear(128, num_labels)

        # @todo should check
        xavier_uniform_(fc1.weight)
        xavier_uniform_(fc2.weight)
        xavier_uniform_(self.fc3.weight)

    def forward(
        self,
        img=None,
        labels=None,
        n_crops=None,
        batch_size=None,
    ):
        ## ALARM
        x = self.visual_features.forward_features(img)
        #x= torch.mean(x,dim=1)
        logits = self.fc3(self.fc2(self.fc1(x)))

        if n_crops is not None:
            logits = logits.view(batch_size, n_crops, -1).mean(1)

        # logits = self.fc(img_feat_)
        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.float())
            outputs = (loss,) + outputs
        return outputs
class ViTransferClassificationLayersWithAttention(nn.Module):
    def __init__(self, num_labels_per_task):
        super().__init__()
        self.num_labels = len(num_labels_per_task)
        # image encoder
        self.visual_features = viTransformer.vit_large_patch32_384(pretrained=True)
        self.attention= Attention(self.visual_features.num_features, self.num_labels)

        self.fc1=nn.Linear(self.visual_features.num_features, self.num_labels)
        #elf.fc2=nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, self.num_labels)
        #self.fc4 = nn.Linear(128, num_labels)

        # @todo should check
        xavier_uniform_(self.fc1.weight)
        #xavier_uniform_(self.fc2.weight)
        #xavier_uniform_(self.fc3.weight)

    def forward(
        self,
        img=None,
        labels=None,
        n_crops=None,
        batch_size=None,
    ):
        ## ALARM
        x = self.visual_features.get_embeddings(img)
        logits = self.attention(x)
        #logits = self.fc1(x)
        if n_crops is not None:
            logits = logits.view(batch_size, n_crops, -1).mean(1)

        # logits = self.fc(img_feat_)
        outputs = (logits,)  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.float())
            outputs = (loss,) + outputs
        return outputs


ClassifierClass={
    "multi-task" : MultiTaskClassificationModel,
    "multi-label" : ViTransferClassification,
}