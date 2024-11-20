from sklearn.model_selection import KFold
import torch


class CrossValidation:
    def __init__(self, number_split, batch_size = 32, shuffle=True):
        self.kFold = kfold = KFold(n_splits = number_split, shuffle=shuffle)
        self.batch_size = batch_size

    def GetSplitDataloader(self, dataset):
        for fold, (train_ids, validation_ids) in enumerate(self.kFold.split(dataset)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
            train_dataloader = torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=self.batch_size, 
                                                sampler=train_subsampler)
            validation_dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=self.batch_size, 
                                                sampler=validation_subsampler)
            yield train_dataloader, validation_dataloader