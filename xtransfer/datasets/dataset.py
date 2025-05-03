import numpy as np
import os
from .utils import transform_options, dataset_options, collate_fn_options
from torch.utils.data import DataLoader
from torchvision import transforms


class DatasetGenerator():
    def __init__(self, train_bs=128, eval_bs=256, seed=0, n_workers=4, 
                 train_d_type='CIFAR10', test_d_type='CIFAR10',
                 train_path='../../datasets/', test_path='../../datasets/',
                 train_tf_op=None, test_tf_op=None, **kwargs):

        np.random.seed(seed)
        self.bd_mode = False
        if train_d_type not in dataset_options:
            print(train_d_type)
            raise('Unknown Dataset')
        elif test_d_type not in dataset_options:
            print(test_d_type)
            raise('Unknown Dataset')

        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers
        self.train_path = train_path
        self.test_path = test_path
        try:
            env_n_workers = os.environ['SLURM_CPUS_PER_TASK']
            if env_n_workers is not None:
                self.n_workers = int(env_n_workers)
            print('setting n_workers to', self.n_workers)
        except:
            print('setting n_workers base on SLURM failed, n_workers is {}'.format(self.n_workers))
        
        if type(train_tf_op) is str:
            train_tf = transform_options[train_tf_op]["train_transform"]
        else:
            train_tf = transform_options[train_tf_op]["train_transform"]
        if type(test_tf_op) is str:
            test_tf = transform_options[test_tf_op]["test_transform"]
        else:
            test_tf = transform_options[test_tf_op]["test_transform"]
        if train_tf is not None:
            train_tf = transforms.Compose(train_tf)
        if test_tf is not None:
            test_tf = transforms.Compose(test_tf)
        
        kwargs['seed'] = seed
        self.train_set = dataset_options[train_d_type](train_path, train_tf, False, kwargs)
        if 'test_bd_d_type' in kwargs or 'test_bd_path' in kwargs:
            self.bd_mode = True
            test_bd_d_type = kwargs['test_bd_d_type']
            test_bd_path = kwargs['test_bd_path']
            self.bd_test_set = dataset_options[test_bd_d_type](test_bd_path, test_tf, True, kwargs)
        else:
            self.bd_test_set = dataset_options[train_d_type](train_path, test_tf, False, kwargs)
        
        self.test_set = dataset_options[test_d_type](test_path, test_tf, True, kwargs)
        self.train_set_length = len(self.train_set)
        self.test_set_length = len(self.test_set)

        if 'train_collate_fn' in kwargs and 'val_collate_fn' in kwargs:
            self.train_collate_fn = collate_fn_options[kwargs['train_collate_fn']['name']](**kwargs['train_collate_fn'])
            self.val_collate_fn = collate_fn_options[kwargs['val_collate_fn']['name']](**kwargs['val_collate_fn'])
        else:
            self.train_collate_fn = None
            self.val_collate_fn = None

    def get_loader(self, train_shuffle=True, drop_last=False, train_sampler=None, test_sampler=None, sampler_bd=None):
        if train_shuffle is False or train_sampler is None:
            train_loader = DataLoader(dataset=self.train_set, pin_memory=True,
                                      batch_size=self.train_bs, drop_last=drop_last,
                                      num_workers=self.n_workers, shuffle=train_shuffle,
                                      collate_fn=self.train_collate_fn)
            test_loader = DataLoader(dataset=self.test_set, pin_memory=True,
                                     batch_size=self.eval_bs, drop_last=drop_last, 
                                     num_workers=self.n_workers, shuffle=False,
                                     collate_fn=self.val_collate_fn)
            bd_loader = DataLoader(dataset=self.bd_test_set, pin_memory=True,
                                   batch_size=self.eval_bs, drop_last=False,
                                   num_workers=self.n_workers,
                                   shuffle=False, collate_fn=self.val_collate_fn)
        else:
            train_loader = DataLoader(dataset=self.train_set, pin_memory=True, 
                                      batch_size=self.train_bs, drop_last=drop_last, 
                                      num_workers=self.n_workers, sampler=train_sampler,
                                      collate_fn=self.train_collate_fn)
            test_loader = DataLoader(dataset=self.test_set, pin_memory=True, 
                                     batch_size=self.eval_bs, drop_last=False, shuffle=False,
                                     num_workers=self.n_workers, sampler=test_sampler,
                                     collate_fn=self.val_collate_fn)
            bd_loader = DataLoader(dataset=self.bd_test_set, pin_memory=True, 
                                   batch_size=self.eval_bs, drop_last=False, 
                                   num_workers=self.n_workers,
                                   sampler=test_sampler if sampler_bd is None else sampler_bd,
                                   collate_fn=self.val_collate_fn)
        return train_loader, test_loader, bd_loader
