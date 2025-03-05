import gc
import json
import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BaseModel
from utils import const, utils

from .dataset import *


not_use_det_model = [
    'SASRec', 'GRU4Rec', 'CL4SRec', 'Query_SeqRec', 'UnifiedSSR'
]

# Author
NEW_EVAL = True

class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument(
            '--num_negs',
            type=int,
            default=5,
            help='num neg samples for each pos sample during training')
        parser.add_argument('--data',
                            type=str,
                            default='Commercial_August_v0',
                            help='Choose a dataset.')

        parser.add_argument('--epoch',
                            type=int,
                            default=1,
                            help='Number of epochs.')
        # parser.add_argument('--test_epoch', type=int, default=-1,
        #                     help='Print test results every test_epoch (-1 means no print).')
        # parser.add_argument('--initial_epoch', type=int, default=0)
        # parser.add_argument('--print_interval', type=int, default=500)

        parser.add_argument('--lr',
                            type=float,
                            default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--lr_scheduler', type=int, default=0)
        parser.add_argument(
            '--patience',
            type=int,
            default=3,
            help=
            'Number of epochs with no improvement after which learning rate will be reduced'
        )
        parser.add_argument(
            '--early_stop',
            type=int,
            default=10,
            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--min_lr', type=float, default=1e-6)
        parser.add_argument('--l2',
                            type=float,
                            default=1e-5,
                            help='Weight decay in optimizer.')

        parser.add_argument('--infoNCE_neg_sample', type=int, default=1024)

        parser.add_argument('--batch_size',
                            type=int,
                            # default=1024,
                            # default=256,
                            # default=64,
                            # default=1,
                            default=8,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size',
                            type=int,
                            # default=512,
                            default=8,
                            help='Batch size during testing.')
        # parser.add_argument('--optimizer', type=str, default='Adam',
        #                     help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument(
            '--num_workers',
            type=int,
            default=10,
            help='Number of processors when prepare batches in DataLoader')
        # parser.add_argument('--pin_memory', type=int, default=1,
        #                     help='pin_memory in DataLoader')

        # parser.add_argument('--topk', type=str, default='1,5,10,20,50',
        #                     help='The number of items recommended to each user.')
        # parser.add_argument('--metric', type=str, default='NDCG,HR',
        #                     help='metrics: NDCG, HR')

        # parser.add_argument('--bias_l2', type=int, default=0)

        return parser

    def __init__(self, model: BaseModel, args) -> None:
        self.args = args

        # self.bias_l2 = args.bias_l2
        self.bias_l2 = 0

        self.num_negs = args.num_negs
        self.data: str = args.data
        self.model = model

        self.epoch = args.epoch
        self.test_epoch = -1  # args.test_epoch
        # self.initial_epoch = args.initial_epoch
        # self.print_interval = args.print_interval
        self.print_interval = 500 * 10

        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.patience = args.patience
        self.min_lr = args.min_lr
        self.l2 = args.l2
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.optimizer_name = 'Adam'  # args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = True  # args.pin_memory

        self.infoNCE_neg_sample = args.infoNCE_neg_sample

        # [int(x) for x in args.topk.split(',')]
        self.topk = [1, 5, 10, 20, 50]

        # [m.strip().upper() for m in args.metric.split(',')]
        self.metrics = ['NDCG', 'HR', 'MRR']

        # early stop based on main_metric
        # self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])
        self.main_metric = 'NDCG@5'
        # logging.info('main metric: {}'.format(self.main_metric))

        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.optimizer: torch.optim.Optimizer = None

        
        self.runner_start = time.time()
        # self.user_vocab = None
        # self.item_vocab = None
        # self.src_session_vocab = None
        # self.datasetParaDict = {
        #     'num_negs': self.num_negs,
        #     "user_vocab": None,
        #     "item_vocab": None,
        #     "src_session_vocab": None
        # }
        # self.query_vocab = None

    def _build_optimizer(self, model: BaseModel):
        if self.bias_l2:
            self.optimizer = eval('torch.optim.{}'.format(
                self.optimizer_name))(model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.l2)
        else:
            self.optimizer = eval('torch.optim.{}'.format(
                self.optimizer_name))(model.customize_parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.l2)

        if self.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.patience,
                min_lr=self.min_lr,
                verbose=True)

    def getDataLoader(self, dataset: BaseDataSet, batch_size: int,
                      shuffle: bool) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=batch_size // self.num_workers + 1,
            worker_init_fn=utils.worker_init_fn,
            persistent_workers=True,
            collate_fn=dataset.collate_batch)
        return dataloader

    def set_dataloader(self):
        self.train_loader = self.getDataLoader(
            self.traindata,
            batch_size=self.batch_size,
            shuffle=const.shuffle_train_data)
        self.val_loader = self.getDataLoader(self.valdata,
                                             batch_size=self.eval_batch_size,
                                             shuffle=False)
        self.test_loader = self.getDataLoader(self.testdata,
                                              batch_size=self.eval_batch_size,
                                              shuffle=False)
        # if True or self.model.query_item_alignment:
            # self.get_query_vocab()
            
        self.InfoNCE_dataloader = self.getDataLoader(
            InfoNCEDataset(),
            batch_size=self.infoNCE_neg_sample,
            shuffle=False)
        
        if self.args.InfoNCE_kw2item_alpha > 0.0:
            self.kw2item_infonce_dataloader = self.getDataLoader(
                KwItemInfoNCEDataset(token_id_all_num=self.model.text_embedding.num_embeddings),
                batch_size=self.args.InfoNCE_kw2item_batchsize,
                shuffle=False)

    def train(self, model: BaseModel):
        # print()
        # random_test_result, _ = self.evaluate(model, 'test')
        # logging.info("Random Test Result:")
        # logging.info(random_test_result)

        # print()
        # logging.info("model fit")
        self._build_optimizer(model)
        torch.autograd.set_detect_anomaly(True) # DBG

        main_metric_results, dev_results = list(), list()
        test_result = None
        for epoch in range(self.epoch):
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("TOTAL TIME: {}".format(utils.format_time(time.time() - self.runner_start)))
            
            if self.args.freeze_text_mapping_epoch > 0:
                if epoch == 0:
                    for name, param in model.named_parameters():
                        if 'text' in name or 'query' in name: 
                            param.requires_grad = False
                elif epoch == self.args.freeze_text_mapping_epoch:
                    for name, param in model.named_parameters():
                        if 'text' in name or 'query' in name: 
                            param.requires_grad = True
                    
            
            epoch_loss = self.train_epoch(epoch, model)

            logging.info("epoch:{} mean loss:{:.4f}".format(epoch, epoch_loss))
            logging.info("TOTAL TIME: {}".format(utils.format_time(time.time() - self.runner_start)))

            dev_result, main_result = self.evaluate(model, 'val', tqdm_desc=f'val epoch {epoch}')
            dev_results.append(dev_result)
            main_metric_results.append(main_result)
            logging.info("Dev Result:")
            logging.info(dev_result)
            if self.lr_scheduler:
                self.scheduler.step(main_result)

            if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                test_result, _ = self.evaluate(model, 'test', tqdm_desc=f'test epoch {epoch}')
                logging.info("Test Result:")
                logging.info(test_result)

            if self.args.force_save_epochs: # break
                if epoch in self.args.force_save_epochs:
                    model.eval()
                    model.save_model(epoch=epoch)
                    test_result, _ = self.evaluate(model, 'test', tqdm_desc=f'test epoch {epoch}')
                    logging.info("Test Result:")
                    logging.info(test_result)
                    if epoch == self.args.force_save_epochs[-1]:
                        break
            else:
                if max(main_metric_results) == main_metric_results[-1]:
                    model.eval()
                    model.save_model(epoch=epoch)
                    test_result, _ = self.evaluate(model, 'test', tqdm_desc=f'test epoch {epoch}')
                    logging.info("Test Result:")
                    logging.info(test_result)

                if self.early_stop > 0 and self.eval_termination(
                        main_metric_results):
                    logging.info('Early stop at %d based on dev result.' %
                                (epoch + 1))
                    break

        best_epoch = main_metric_results.index(max(main_metric_results))
        print()
        logging.info("Best Dev Result at epoch:{}".format(best_epoch))
        logging.info(dev_results[best_epoch])
        logging.info("Test Result:")
        logging.info(test_result)
        with open(f'{self.args.model_path}/result.json', 'w')as f:
            json.dump(dict(best_val_epoch=best_epoch, total_time=utils.format_time(time.time() - self.runner_start),
                           val_results=dev_results,test_result=test_result), f)
            
        model.load_model(epoch=best_epoch)

        test_result, _ = self.evaluate(model, 'test', tqdm_desc=f'test loaded epoch {best_epoch}')
        print()
        logging.info("Load Test Result:")
        logging.info(test_result)

    def train_epoch(self, epoch: int, model: BaseModel):
        model.train()
        print()
        logging.info("Epoch: {}".format(epoch))

        if model.query_item_alignment:
            InfoNCE_iterator = iter(self.InfoNCE_dataloader)
        if self.args.InfoNCE_kw2item_alpha > 0.0:
            kw2item_infonce_iterator = iter(self.kw2item_infonce_dataloader)    
        

        # logging.info("start training")
        loss_list = []
        loss_dict = {}
        start = time.time()
        for step, batch in enumerate(tqdm(self.train_loader,desc=f"Train epoch {epoch}",
                                          mininterval=5)):
            if model.query_item_alignment:
                align_dict = next(InfoNCE_iterator)
                batch.update(align_dict)
            if self.args.InfoNCE_kw2item_alpha > 0.0:
                align_dict = next(kw2item_infonce_iterator)
                batch.update(align_dict)
                
           
            loss = model.loss(utils.batch_to_gpu(batch, model.device))

            total_loss = loss['total_loss']

            self.optimizer.zero_grad()

            # 
            if self.args.model in not_use_det_model:
                torch.use_deterministic_algorithms(False)
                total_loss.backward()
                torch.use_deterministic_algorithms(True)
            else:
                total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(
            #     parameters=model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

            for k, v in loss.items():
                if k in loss_dict.keys():
                    loss_dict[k].append(v.item())
                else:
                    loss_dict[k] = [v.item()]

            loss_list.append(total_loss.item())

            if step > 0 and step % self.print_interval == 0:
                logging.info("epoch:{:d} step:{:d} time:{} {}".format(
                    epoch, step,
                    utils.format_time(time.time() - start), " ".join([
                        "{}:{:.4f}".format(k,
                                           np.mean(v).item())
                        for k, v in loss_dict.items()
                    ])))

            # if hasattr(model, 'behavior_transformer'):
            # logging.info("{}".format(
            #     model.behavior_transformer.transformer_blocks[0].attention.linear_layers[0].max()))
            # logging.info("{}".format(
            #     model.behavior_transformer.transformer_blocks[0].attention.linear_layers[1].max()))
            # logging.info("{}".format(
            #     model.behavior_transformer.transformer_blocks[0].attention.linear_layers[2].max()))

            # if abs(model.behavior_transformer.transformer_blocks[0].attention.linear_layers[0].weight.max()) < 1e-7:
            #     logging.info(
            #         "berak at epoch:{} step:{}".format(epoch, step))
            #     break

            # if abs(model.behavior_transformer.transformer_blocks[0].attention.W1.max()) < 1e-7:
            #     logging.info(
            #         "berak at epoch:{} step:{}".format(epoch, step))
            #     break

            # if step > 0 and step % self.print_interval == 0:
            #     logging.info("epoch:{:d} step:{:d} loss:{:.4f} time:{}s".format(
            #         epoch, step, np.mean(loss_list).item(), time.time() - start))
        logging.info("train_epoch total time: {}".format(utils.format_time(time.time() - start)))

        return np.mean(loss_list).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: List[int],
                        metrics: List[str]) -> Dict[str, float]:
        raise NotImplementedError

    @staticmethod
    @torch.no_grad()
    def predict(model: BaseModel, test_loader: DataLoader, print_interval, tqdm_desc='predict'):
        raise NotImplementedError

    def evaluate(self, model: BaseModel, mode: str, tqdm_desc):
        if mode == 'val':
            predictions = self.predict(model,
                                       self.val_loader,
                                       print_interval=self.print_interval, tqdm_desc=tqdm_desc)
        elif mode == 'test':
            predictions = self.predict(model,
                                       self.test_loader,
                                       print_interval=self.print_interval, tqdm_desc=tqdm_desc)
        else:
            raise ValueError('test set error')
        results = self.evaluate_method(predictions, self.topk, self.metrics)
        return utils.format_metric(results), results[self.main_metric]

    def test(self, model: BaseModel, mode: str, tqdm_desc):
        if mode == 'val':
            predictions = self.predict(model,
                                       self.val_loader,
                                       print_interval=self.print_interval, tqdm_desc=tqdm_desc)
        elif mode == 'test':
            predictions = self.predict(model,
                                       self.test_loader,
                                       print_interval=self.print_interval, tqdm_desc=tqdm_desc)
        else:
            raise ValueError('test set error')
        results = self.evaluate_method(predictions, self.topk, self.metrics)
        return results

    def build_dataset(self):
        pass


class RecRunner(BaseRunner):
    def __init__(self, model, args) -> None:
        super().__init__(model, args)
        self.build_dataset()
        self.set_dataloader()

    def build_dataset(self):
        super().build_dataset()
        self.traindata = RecDataSet(train_mode='train',
                                    args=self.args, 
        )
        self.valdata = RecDataSet(train_mode='val',
                                  args=self.args,                  
        )
        self.testdata = RecDataSet(train_mode='test',
                                   args=self.args,                  
        )

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: List[int],
                        metrics: List[str]) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()
        
        if NEW_EVAL:
            predictions = np.concatenate((predictions[:,1:],predictions[:,:1]), axis=1)
            gt_idx = predictions.shape[1]-1
        else:
            gt_idx = 0
        sort_idx = (-predictions).argsort(axis=1)
        gt_rank = np.argwhere(sort_idx == gt_idx)[:, 1] + 1
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                elif metric == 'MRR':
                    evaluations[key] = (hit / gt_rank).mean() 
                    # evaluations[key] = np.where(hit > 0, 1.0 / gt_rank, 0.0).mean()
                else:
                    raise ValueError(
                        'Undefined evaluation metric: {}.'.format(metric))
        return evaluations

    @staticmethod
    @torch.no_grad()
    def predict(model: BaseModel, test_loader: DataLoader, print_interval, tqdm_desc='predict'):
        model.eval()
        predictions = list()

        # tqdm_ = tqdm(test_loader)
        start = time.time()
        for step, batch in enumerate(tqdm(test_loader,desc=tqdm_desc,mininterval=5)):
            prediction = model.predict(utils.batch_to_gpu(batch, model.device))
            predictions.extend(prediction.cpu().data.numpy())

            if step > 0 and step % print_interval == 0:
                logging.info("step:{:d} time:{}".format(
                    step,
                    utils.format_time(time.time() - start)))

        logging.info("model evaluate time used:{}".format(utils.format_time(time.time() -
                                                           start)))

        # torch.cuda.empty_cache()

        predictions = utils.pad_and_stack(predictions,pad_value=-np.inf)

        return predictions 


class SrcRunner(BaseRunner):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.build_dataset()
        self.set_dataloader()

    def build_dataset(self):
        super().build_dataset()
        self.traindata = SrcDataSet(train_mode='train',
                                    args=self.args,
                                    # **self.model.ParameterDict
        )
        self.valdata = SrcDataSet(train_mode='val',
                                  args=self.args,
                                #   **self.model.ParameterDict
        )
        self.testdata = SrcDataSet(train_mode='test',
                                   args=self.args,
            
        )

    @staticmethod
    def evaluate_method(predictions, topk: List[int],
                        metrics: List[str]) -> Dict[str, float]:
        return RecRunner.evaluate_method(predictions, topk, metrics)

    @staticmethod
    @torch.no_grad()
    def predict(model: BaseModel, test_loader: DataLoader, print_interval,tqdm_desc='predict'):
        return RecRunner.predict(model, test_loader, print_interval,tqdm_desc)

class SarRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--src_loss_weight', type=float, default=1.0, help='src loss weight, only used when SarRunner')

        return BaseRunner.parse_runner_args(parser)

    def __init__(self, model, args) -> None:
        super().__init__(model, args)
        self.build_dataset()
        self.set_dataloader()

        self.src_loss_weight = args.src_loss_weight

    def set_dataloader(self):
        self.rec_train_loader = self.getDataLoader(
            self.traindata['rec'],
            batch_size=self.batch_size,
            shuffle=const.shuffle_train_data)
        self.rec_val_loader = self.getDataLoader(
            self.valdata['rec'],
            batch_size=self.eval_batch_size,
            shuffle=False)
        self.rec_test_loader = self.getDataLoader(
            self.testdata['rec'],
            batch_size=self.eval_batch_size,
            shuffle=False)

        src_train_batch_size = len(self.traindata['src']) // (
            len(self.traindata['rec']) // self.batch_size + 1) + 1
        logging.info('SarRunner: src train batch size:{}'.format(src_train_batch_size))
        self.src_train_loader = self.getDataLoader(
            self.traindata['src'],
            batch_size=src_train_batch_size,
            shuffle=const.shuffle_train_data)
        self.src_val_loader = self.getDataLoader(
            self.valdata['src'],
            batch_size=self.eval_batch_size,
            shuffle=False)
        self.src_test_loader = self.getDataLoader(
            self.testdata['src'],
            batch_size=self.eval_batch_size,
            shuffle=False)

    def build_dataset(self):
        super().build_dataset()
        self.traindata = {
            "rec":
            RecDataSet(train_mode='train',
                       args=self.args,

                       ),
            "src":
            SrcDataSet(train_mode='train',
                       args=self.args,

                       )
        }
        self.valdata = {
            "rec":
            RecDataSet(train_mode='val',
                       args=self.args,

                       ),
            "src":
            SrcDataSet(train_mode='val',
                       args=self.args,

                       )
        }
        self.testdata = {
            "rec":
            RecDataSet(train_mode='test',
                       args=self.args,

                       ),
            "src":
            SrcDataSet(train_mode='test',
                       args=self.args,

                       )
        }

    def train_epoch(self, epoch: int, model: BaseModel):
        model.train()
        print()
        logging.info("Epoch: {}".format(epoch))

        if model.query_item_alignment:
            InfoNCE_iterator = iter(self.InfoNCE_dataloader)

        src_iterator = iter(self.src_train_loader)

        # logging.info("start training")
        loss_list = []
        loss_dict = {"rec": {}, "src": {}}
        start = time.time()
        for step, rec_batch in enumerate(tqdm(self.rec_train_loader,desc=f'Train epoch {epoch}',
                                              mininterval=5)):
            try:
                src_batch = next(src_iterator)
            except StopIteration:
                src_iterator = iter(self.src_train_loader)
                src_batch = next(src_iterator)

            if model.query_item_alignment:
                align_dict = next(InfoNCE_iterator)
                rec_batch.update(align_dict)
                src_batch.update(align_dict)

            rec_loss = model.loss(utils.batch_to_gpu(rec_batch, model.device))
            src_loss = model.src_loss(
                utils.batch_to_gpu(src_batch, model.device))

            for k in rec_loss.keys():
                if k in loss_dict['rec'].keys():
                    loss_dict['rec'][k].append(rec_loss[k].item())
                    loss_dict['src'][k].append(src_loss[k].item())
                else:
                    loss_dict['rec'][k] = [rec_loss[k].item()]
                    loss_dict['src'][k] = [src_loss[k].item()]

            # total_loss = rec_loss['total_loss'] + src_loss['total_loss']
            total_loss = rec_loss['total_loss'] + \
                src_loss['total_loss'] * self.src_loss_weight

            self.optimizer.zero_grad()

            # 
            if self.args.model in not_use_det_model:
                torch.use_deterministic_algorithms(False)
                total_loss.backward()
                torch.use_deterministic_algorithms(True)
            else:
                total_loss.backward()

            # total_loss.backward()
            self.optimizer.step()
            loss_list.append(total_loss.item())

            if step > 0 and step % self.print_interval == 0:
                logging.info(
                    "epoch:{:d} step:{:d} time:{:.2f}s |rec {}| |src {}|".
                    format(
                        epoch, step,
                        utils.format_time(time.time() - start), " ".join([
                            "{}:{:.4f}".format(k,
                                               np.mean(v).item())
                            for k, v in loss_dict['rec'].items()
                        ]), " ".join([
                            "{}:{:.4f}".format(k,
                                               np.mean(v).item())
                            for k, v in loss_dict['src'].items()
                        ])))
        logging.info("train_epoch total time: {}".format(utils.format_time(time.time() - start)))

        return np.mean(loss_list).item()

    def evaluate(self, model: BaseModel, mode: str, tqdm_desc):
        if mode == 'val':
            rec_predictions = RecRunner.predict(
                model, self.rec_val_loader, print_interval=self.print_interval,tqdm_desc=tqdm_desc)
            src_predictions = SrcRunner.predict(
                model, self.src_val_loader, print_interval=self.print_interval,tqdm_desc=tqdm_desc)
        elif mode == 'test':
            rec_predictions = RecRunner.predict(
                model,
                self.rec_test_loader,
                print_interval=self.print_interval,tqdm_desc=tqdm_desc)
            src_predictions = SrcRunner.predict(
                model,
                self.src_test_loader,
                print_interval=self.print_interval,tqdm_desc=tqdm_desc)
        else:
            raise ValueError('test set error')
        rec_results = RecRunner.evaluate_method(rec_predictions, self.topk,
                                                self.metrics)
        src_results = SrcRunner.evaluate_method(src_predictions, self.topk,
                                                self.metrics)

        results = {
            "rec": utils.format_metric(rec_results),
            "src": utils.format_metric(src_results)
        }
        return results, (rec_results[self.main_metric] +
                         src_results[self.main_metric]) / 2.0

    def test(self, model: BaseModel, mode: str, tqdm_desc):
        if mode == 'val':
            rec_predictions = RecRunner.predict(
                model, self.rec_val_loader, print_interval=self.print_interval,tqdm_desc=tqdm_desc)
            src_predictions = SrcRunner.predict(
                model, self.src_val_loader, print_interval=self.print_interval,tqdm_desc=tqdm_desc)
        elif mode == 'test':
            rec_predictions = RecRunner.predict(
                model,
                self.rec_test_loader,
                print_interval=self.print_interval,tqdm_desc=tqdm_desc)
            src_predictions = SrcRunner.predict(
                model,
                self.src_test_loader,
                print_interval=self.print_interval,tqdm_desc=tqdm_desc)
        else:
            raise ValueError('test set error')
        rec_results = RecRunner.evaluate_method(rec_predictions, self.topk,
                                                self.metrics)
        src_results = SrcRunner.evaluate_method(src_predictions, self.topk,
                                                self.metrics)

        results = {"rec": rec_results, "src": src_results}
        return results

