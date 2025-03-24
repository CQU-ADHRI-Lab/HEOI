import torch.utils.data
class CustomDatasetDataLoader:
    def __init__(self, opt, is_for_train=True):  # opt short for options
        self._opt = opt
        self._is_for_train = is_for_train
        self._num_threds = opt.n_threads_train if is_for_train \
            else opt.n_threads_test
        self._create_dataset()

    def _create_dataset(self):
        self._dataset = DatasetFactory.get_by_name(self._opt.dataset_mode,
                                                   self._opt,
                                                   self._is_for_train)
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self._opt.batch_size,
            # shuffle=False,
            shuffle=self._is_for_train,
            num_workers=int(self._num_threds),
            drop_last=True)

    def load_data(self):
        return self._dataloader

    def __len__(self):
        return len(self._dataset)


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'CAD':
            from data_n.dataset_CAD import CustomDataset
            dataset = CustomDataset(opt, is_for_train)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset
