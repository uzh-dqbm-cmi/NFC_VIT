import torch.nn as nn
import torchvision
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.nn.functional as F
from Model import viTransformer


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
        self.visual_features = MultiTaskDenseNet121(num_labels_per_task)

    def forward(
        self,
        img=None,
        labels=None,
        n_crops=None,
        batch_size=None,
    ):

        logits = self.visual_features(img)
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
        self.visual_features = viTransformer.vit_large_patch16_224(pretrained=True, num_classes=self.num_labels)

    def forward(
        self,
        img=None,
        labels=None,
        n_crops=None,
        batch_size=None,
    ):
        logits = self.visual_features(img)
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
    "multi-task":MultiTaskClassificationModel,
    "multi-label":ViTransferClassification
}