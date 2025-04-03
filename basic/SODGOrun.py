from data_loader import *
from model import *
import traceback
import os
from ordered_set import OrderedSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import collections

import torch.cuda.amp as amp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

class Runner(object):

    def load_data(self):
        dataset_ent2id_path = ('{}/data/{}/go_id_dict_384.txt'.format(self.p.rootpath, self.p.dataset))
        dataset_rel2id_path = ('{}/data/{}/go_relation_dict.txt'.format(self.p.rootpath,self.p.dataset))

        if os.path.exists(dataset_ent2id_path):
            self.ent2id, self.rel2id, self.id2ent, self.id2rel = self.load_or_create_mappings(dataset_ent2id_path, dataset_rel2id_path)
        else:
            ent_set, rel_set = OrderedSet(), OrderedSet()
            for split in ['train', 'test', 'valid']:
                for line in open('{}/data/{}/{}.txt'.format(self.p.rootpath, self.p.dataset, split)):
                    sub, rel, obj = map(str, line.strip().split('\t'))
                    ent_set.add(sub)
                    rel_set.add(rel)
                    ent_set.add(obj)
            self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
            self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
            self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})
            self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
            self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        # self.node_attributes, self.namespace2id = load_node_attributes('data/prior/GO_IDs_and_Namespaces.csv', self.ent2id)
        self.node_attributes, self.namespace2id, self.init_goemb = load_node_attributes('{}/data/prior/GO_IDs_Namespaces_Embedding.csv'.format(self.p.rootpath), self.ent2id)
        self.rel_strength = load_relationship_strengths('{}/data/prior/namespace_relationship_counts.csv'.format(self.p.rootpath), self.namespace2id, self.rel2id)

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        self.logger.info('num_ent {} num_rel {}'.format(self.p.num_ent, self.p.num_rel))
        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in open('{}/data/{}/{}.txt'.format(self.p.rootpath,self.p.dataset, split)):
                sub, rel, obj = line.strip().split('\t')
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))
                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        if self.p.score_func == 'similarity':
            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        else:
            if self.p.strategy == 'one_to_n':
                for (sub, rel), obj in self.sr2o.items():
                    self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
            else:
                for sub, rel, obj in self.data['train']:
                    rel_inv = rel + self.p.num_rel
                    sub_samp = len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
                    sub_samp = np.sqrt(1 / sub_samp)

                    self.triples['train'].append({'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': sub_samp})
                    self.triples['train'].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)], 'sub_samp': sub_samp})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})
            self.logger.info('{}_{} num is {}'.format(split, 'tail', len(self.triples['{}_{}'.format(split, 'tail')])))
            self.logger.info('{}_{} num is {}'.format(split, 'head', len(self.triples['{}_{}'.format(split, 'head')])))

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p, self.node_attributes, self.rel_strength),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.test_batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.test_batch_size),
        }
        self.logger.info('num_ent {} num_rel {}\n'.format(self.p.num_ent, self.p.num_rel))
        self.logger.info('train set num is {}\n'.format(len(self.triples['train'])))
        self.logger.info('{}_{} num is {}\n'.format('test', 'tail', len(self.triples['{}_{}'.format('test', 'tail')])))
        self.logger.info('{}_{} num is {}\n'.format('valid', 'tail', len(self.triples['{}_{}'.format('valid', 'tail')])))
        self.edge_index, self.edge_type = self.construct_adj()
        #保存self.edge_index, self.edge_type
        # torch.save(self.edge_index, 'attention_analyse/edge_index.pt')
        # torch.save(self.edge_type, 'attention_analyse/edge_type.pt')

    def load_or_create_mappings(dataset_path, go_id_dict_path=None, go_relation_dict_path=None):
        ent_set, rel_set = collections.OrderedDict(), collections.OrderedDict()

        if os.path.exists(go_id_dict_path):
            with open(go_id_dict_path, 'r') as file:
                ent2id = {line.split('\t')[0]: int(line.split('\t')[1].strip()) for line in file.readlines()}
            with open(go_relation_dict_path, 'r') as file:
                rel2id = {line.split('\t')[0]: int(line.split('\t')[1].strip()) for line in file.readlines()}
        else:
            for split in ['train', 'test', 'valid']:
                with open(os.path.join(dataset_path, f"{split}.txt")) as file:
                    for line in file:
                        sub, rel, obj = map(str, line.strip().split('\t'))
                        ent_set[sub] = None
                        rel_set[rel] = None
                        ent_set[obj] = None
            ent2id = {ent: idx for idx, ent in enumerate(ent_set.keys())}
            rel2id = {rel: idx for idx, rel in enumerate(rel_set.keys())}
            rel2id.update({rel + '_reverse': idx + len(rel2id) for idx, rel in enumerate(rel_set.keys())})

        rel2id.update({rel + '_reverse': idx + len(rel2id) for rel, idx in rel2id.items()})
        id2ent = {idx: ent for ent, idx in ent2id.items()}
        id2rel = {idx: rel for rel, idx in rel2id.items()}

        return ent2id, rel2id, id2ent, id2rel

    def construct_adj(self):
        edge_index, edge_type = [], []
        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir, self.p.rootpath)
        self.inp_drop = torch.nn.Dropout(self.p.iinp_drop)
        self.logger.info(vars(self.p))
        self.writer = SummaryWriter(log_dir=self.p.log_dir + 'tensorboard')
        pprint(vars(self.p))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("GPU is available")
        else:
            self.device = torch.device('cpu')
            print("GPU not available, CPU used")
        self.load_data()
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer, self.optimizer_mi = self.add_optimizer(self.model)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, patience=5, min_lr=1e-5)


    def add_model(self, model, score_func):
        model_name = '{}_{}'.format(model, score_func)
        if model_name.lower() == 'sodgo_transe':
            model = SODGO_TransE(self.edge_index, self.edge_type, params=self.p
                                     , node_attributes=self.node_attributes, rel_strength=self.rel_strength
                                     , init_goemb=self.init_goemb)
        # elif model_name.lower() == 'sodgo_interacte':
        #     model = sodgo_InteractE(self.edge_index, self.edge_type, params=self.p
        #                                 , node_attributes=self.node_attributes, rel_strength=self.rel_strength
        #                                 , init_goemb=self.init_goemb)
        # elif model_name.lower() == 'sodgo_similarity':
        #     model = sodgo_Similarity(self.edge_index, self.edge_type, params=self.p
        #                                  , node_attributes=self.node_attributes, rel_strength=self.rel_strength
        #                                  , init_goemb=self.init_goemb)
            model.to(self.device)
        return model

    # def add_optimizer(self, model):
    #     if self.p.mi_train and self.p.mi_method.startswith('club'):
    #         mi_disc_params = list(map(id, model.mi_Discs.parameters()))
    #         rest_params = filter(lambda x: id(x) not in mi_disc_params, model.parameters())
    #         for m in model.mi_Discs.modules():
    #             self.logger.info(m)
    #         for name, parameters in model.named_parameters():
    #             print(name, ':', parameters.size())
    #         return torch.optim.Adam(rest_params, lr=self.p.lr, weight_decay=self.p.l2), torch.optim.Adam(model.mi_Discs.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
    #     else:
    #         return torch.optim.Adam(model.parameters(), lr=self.p.lr, weight_decay=self.p.l2), None
    def add_optimizer(self, model):
        # 定义一些基本的超参数，例如学习率和权重衰减
        lr = self.p.lr
        weight_decay = self.p.l2
        optimizer_type = self.p.optimizer_type  # 超参数控制优化器类型，比如 'adam', 'sgd', 'rmsprop'

        if self.p.mi_train and self.p.mi_method.startswith('club'):
            mi_disc_params = list(map(id, model.mi_Discs.parameters()))
            rest_params = filter(lambda x: id(x) not in mi_disc_params, model.parameters())

            # 根据超参数选择优化器
            optimizer_main = self._get_optimizer(optimizer_type, rest_params, lr, weight_decay)
            optimizer_mi = self._get_optimizer(optimizer_type, model.mi_Discs.parameters(), lr, weight_decay)

            return optimizer_main, optimizer_mi
        else:
            # 为所有参数选择优化器
            optimizer_main = self._get_optimizer(optimizer_type, model.parameters(), lr, weight_decay)
            return optimizer_main, None

    def _get_optimizer(self, optimizer_type, params, lr, weight_decay):
        """根据传入的类型和参数选择相应的优化器"""
        if optimizer_type == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def read_batch(self, batch, split):
        if split == 'train':
            if self.p.strategy == 'one_to_x':
                triple, label, neg_ent, sub_samp = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, neg_ent, sub_samp
            else:
                triple, label,orig_label = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, orig_label, None, None
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mrr': self.best_val_mrr,
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'args': vars(self.p)
        }
        torch.save(checkpoint, save_path)

    def save_embeddings(self, embeddings, dataset_name, embed_dim, embed_type):
        if embed_type == 'best_valid_mrr':
            filename = f"{dataset_name}_best_valid_mrr_{embed_dim}.txt"
        elif embed_type == 'min_lldloss':
            filename = f"{dataset_name}_min_lldloss_{embed_dim}.txt"
        else:
            raise ValueError("embed_type must be either 'best_accuracy' or 'min_loss'")

        with open(filename, 'w', encoding='utf-8') as f:
            num_entities = embeddings.shape[0]
            f.write(f"{num_entities} {embed_dim}\n")
            for index, entity_embedding in enumerate(embeddings):
                emb_flattened = entity_embedding.flatten()
                emb_str = ' '.join(f"{val:.6e}" for val in emb_flattened.tolist())
                f.write(f"{index} {emb_str}\n")
        print(f"Embeddings saved to {filename}")

    def load_model(self, load_path):
        try:
            checkpoint = torch.load(load_path, weights_only=True)
        except TypeError:
            checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_mrr = checkpoint['best_val_mrr']
        self.best_val = checkpoint['best_val']

    def evaluate(self, split, epoch):
        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)

        res_mrr = '\n\tMRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mrr'], results['right_mrr'], results['mrr'])
        res_mr = '\tMR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mr'], results['right_mr'], results['mr'])
        res_hit1 = '\tHit-1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@1'], results['right_hits@1'], results['hits@1'])
        res_hit3 = '\tHit-3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@3'], results['right_hits@3'], results['hits@3'])
        res_hit10 = '\tHit-10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_hits@10'], results['right_hits@10'], results['hits@10'])
        log_res = res_mrr + res_mr + res_hit1 + res_hit3 + res_hit10

        if (epoch + 1) % 5 == 0 or split == 'test':
            self.logger.info('[Evaluating Epoch {} {}]: {}'.format(epoch, split, log_res))
        else:
            self.logger.info('[Evaluating Epoch {} {}]: {}'.format(epoch, split, res_mrr))

        return results

    def predict(self, split='valid', mode='tail_batch'):
        self.model.eval()
        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            total_correct = 0
            total_samples = 0
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred, _, _, _ = self.model.forward(sub, rel, None, None, split)
                # pred, _, all_ent = self.model.forward(sub, rel, None, split)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get('hits@{}'.format(k + 1), 0.0)
                _, predicted_obj = torch.max(pred, dim=1)
                correct_predictions = (predicted_obj == obj).sum().item()
                total_correct += correct_predictions
                total_samples += pred.size(0)
            accuracy = total_correct / total_samples
            results['accuracy'] = accuracy
            self.logger.info('[{} {}]\tFinal Accuracy: {:.4f}'.format(split.title(), mode.title(), accuracy))

        return results


    def run_epoch(self, epoch, val_mrr=0):
        self.model.train()
        losses = []
        losses_train = []
        corr_losses = []
        lld_losses = []
        train_iter = iter(self.data_iter['train'])
        # 初始化 GradScaler
        scaler = amp.GradScaler()

        # 使用 tqdm 显示进度条
        with tqdm(total=len(self.data_iter['train']), desc=f"Epoch {epoch + 1}/{self.p.max_epochs}",
                  unit='batch') as pbar:
            for step, batch in enumerate(train_iter):
                self.optimizer.zero_grad()
                if self.p.mi_train and self.p.mi_method.startswith('club'):
                    self.model.mi_Discs.eval()
                sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')

                # 使用 autocast 进行混合精度计算
                with amp.autocast():
                    pred, corr, all_ent, analy_alpha = self.model.forward(sub, rel, None, neg_ent, 'train')
                    loss = self.model.loss(pred, label)
                    if self.p.mi_train:
                        losses_train.append(loss.item())
                        loss = loss + self.p.alpha * corr
                        corr_losses.append(corr.item())

                # 使用 scaler 来缩放梯度
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                losses.append(loss.item())

                # 更新进度条
                pbar.update(1)

                # start to compute mi_loss
                if self.p.mi_train and self.p.mi_method.startswith('club'):
                    for i in range(self.p.mi_epoch):
                        self.model.mi_Discs.train()
                        with amp.autocast():
                            lld_loss = self.model.lld_best(sub, rel)
                        self.optimizer_mi.zero_grad()
                        scaler.scale(lld_loss).backward()
                        scaler.step(self.optimizer_mi)
                        scaler.update()
                        lld_losses.append(lld_loss.item())

                if step % 100 == 0:
                    if self.p.mi_train:
                        self.logger.info(
                            '[E:{}| {}]: total Loss:{:.5}, Train Loss:{:.5}, Corr Loss:{:.5}, Val MRR:{:.5}\t{}'.format(
                                epoch, step, np.mean(losses),
                                np.mean(losses_train), np.mean(corr_losses),
                                self.best_val_mrr,
                                self.p.name))
                    else:
                        self.logger.info(
                            '[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses),
                                                                                      self.best_val_mrr,
                                                                                      self.p.name))

        loss = np.mean(losses_train) if self.p.mi_train else np.mean(losses)
        if self.p.mi_train:
            loss_corr = np.mean(corr_losses)
            if self.p.mi_method.startswith('club') and self.p.mi_epoch == 1:
                loss_lld = np.mean(lld_losses)
                return all_ent, loss, loss_corr, loss_lld, analy_alpha
            return all_ent, loss, loss_corr, 0., analy_alpha
        return all_ent, loss, 0., 0., analy_alpha

    def fit(self):
        try:
            self.best_val_mrr, self.best_val, self.best_epoch, val_mrr, self.min_lldloss = 0., {}, 0, 0., 10.
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join('{}/checkpoints'.format(self.p.rootpath),  'checkpoints', f'model_{current_time}.pkl')
            # 确保 'checkpoints' 目录存在
            checkpoints_dir = os.path.dirname(save_path)
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)

            print(f"Model will be saved to: {save_path}")
            if self.p.restore:
                self.load_model(save_path)
                self.logger.info('Successfully Loaded previous model')
            val_results = {}
            val_results['mrr'] = 0
            kill_cnt = 0
            for epoch in range(self.p.max_epochs):
                all_ent, train_loss, corr_loss, lld_loss, analy_alpha = self.run_epoch(epoch)
                val_results = self.evaluate('valid', epoch)
                self.scheduler.step(val_results['mrr'])
                if val_results['mrr'] >= self.best_val_mrr:
                    self.best_val = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.logger.info('Successfully updated model')
                    kill_cnt = 0
                    self.save_embeddings(all_ent, self.p.dataset, self.p.gcn_dim, 'best_valid_mrr')
                    self.logger.info('Successfully updated best_valid_mrr embeddings')
                    np.save("alpha_tensor.npy", analy_alpha.detach().cpu().numpy())
                    self.logger.info('Successfully updated alpha tensor')
                if lld_loss < self.min_lldloss:
                    self.min_lldloss = lld_loss
                    self.save_embeddings(all_ent, self.p.dataset, self.p.gcn_dim, 'min_lldloss')
                    self.logger.info('Successfully updated min_lldloss embeddings')
                else:
                    kill_cnt += 1
                    if kill_cnt % 5 == 0 and self.p.gamma > self.p.max_gamma:
                        self.p.gamma -= 5
                        self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > self.p.early_stop:
                        self.logger.info("Early Stopping!!")
                        break
                if self.p.mi_train:
                    if self.p.mi_method == 'club_s' or self.p.mi_method == 'club_b':
                        self.logger.info('[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, lld loss :{:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss, corr_loss, lld_loss, self.best_val_mrr))
                    else:
                        self.logger.info('[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss, corr_loss, self.best_val_mrr))
                else:
                    self.logger.info('[Epoch {}]: Training Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))
            self.logger.info('Loading best model, Evaluating on Test data')
            self.load_model(save_path)
            test_results = self.evaluate('test', self.best_epoch)
        except Exception as e:
            self.logger.debug("%s____%s\n"
                              "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('-rootpath', dest='rootpath', default='/public/home/lgl_sd/code/fyx/similarity_GO/basic/', help='Root path')
    parser.add_argument('-rootpath', dest='rootpath', default='./', help='Root path')
    parser.add_argument('-name', default='SODGO', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='GO_basic_triplet', help='Dataset to use, default: GO_basic_triplet')
    parser.add_argument('-model', dest='model', default='sodgo', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='transe', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='cross', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-batch', dest='batch_size', default=2048, type=int, help='Batch size')
    parser.add_argument('-test_batch', dest='test_batch_size', default=2048, type=int, help='Batch size of valid and test data')
    parser.add_argument('-gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('-init_dim', dest='init_dim', default=100, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', dest='embed_dim', default=200, type=int, help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.4, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-logdir', dest='log_dir', default='log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='config/', help='Config directory')
    parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('-neg_num', dest="neg_num", default=1000, type=int, help='Number of negative samples to use for loss calculation')
    parser.add_argument("-strategy", type=str, default='one_to_n', help='Training strategy to use')
    parser.add_argument('-form', type=str, default='plain', help='The reshaping form to use')
    parser.add_argument('-head_num', dest="head_num", default=8, type=int, help="Number of attention heads")
    parser.add_argument('-num_factors', dest="num_factors", default=3, type=int, help="Number of factors")
    parser.add_argument('-alpha', dest="alpha", default=1e-1, type=float, help='Dropout for Feature')
    parser.add_argument('-mi_train', dest='mi_train', default='mi_drop', action='store_true', help='whether to disentangle')
    parser.add_argument('-early_stop', dest="early_stop", default=200, type=int, help="number of early_stop")
    parser.add_argument('-no_act', dest='no_act', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-no_enc', dest='no_enc', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-mi_method', dest='mi_method', default='club_b', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-att_mode', dest='att_mode', default='dot_weight', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_epoch', dest="mi_epoch", default=1, type=int, help="Number of MI_Disc training times")
    parser.add_argument('-score_method', dest='score_method', default='dot_rel', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-score_order', dest='score_order', default='after', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-mi_drop', dest='mi_drop', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-max_gamma', type=float, default=5.0, help='Margin')
    parser.add_argument('-fix_gamma', dest='fix_gamma', action='store_true', help='whether to use non_linear function')
    parser.add_argument('-init_gamma', type=float, default=9.0, help='Margin')
    parser.add_argument('-gamma_method', dest='gamma_method', default='norm', help='Composition Operation to be used in RAGAT')
    parser.add_argument('-optimizer_type', type=str, default='adam', help='Optimizer type')
    parser.add_argument('-beta', type=float, default=1e-1, help='Dropout for Feature')
    parser.add_argument('-direct_init', dest='direct_init', action='store_true', help='whether to use direct_init')
    args = parser.parse_args()

    if not args.restore:
        args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = Runner(args)
    model.fit()
