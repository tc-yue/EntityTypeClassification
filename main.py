# -*- coding: utf-8 -*-
# @Time    : 2019/6/16 22:25
# @Author  : Tianchiyue
# @File    : main.py
# @Software: PyCharm


import argparse
import torch.nn.functional as F
from utils import *
import json
import os
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForTokenClassification, BertConfig, BertForMaskedLM, BertModel, \
    BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from model import AttentiveLSTM
from metric import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--seed_num', default=147, type=int)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--valid_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--embedding_data_path', type=str, default=None)

    parser.add_argument('--bert_config_path', type=str, default='bert_chinese/bert_config.json')
    parser.add_argument('--bert_model_path', type=str, default='bert_chinese/pytorch_model.bin')

    parser.add_argument('--model', default='cnn', type=str)
    parser.add_argument('--attention_mode', default='bilinear', type=str)
    parser.add_argument('-e', '--epochs', default=10, type=int)

    parser.add_argument('-d', '--dropout', default=0.2, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('--train_batch_size', default=32, type=int, help='valid/test batch size')
    parser.add_argument('--valid_batch_size', default=64, type=int, help='valid/test batch size')
    parser.add_argument('--test_batch_size', default=64, type=int, help='valid/test batch size')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_prob', default=0.2, type=float)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--valid_step', default=1000, type=int)

    parser.add_argument('--vocab_size', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_labels', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--embedding_dim', type=int, default=300, help='DO NOT MANUALLY SET')
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--trainable_embedding', action='store_true')
    parser.add_argument('--use_crf', action='store_true')
    parser.add_argument('--weighted', action='store_true')

    return parser.parse_args(args)


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def get_dataloader(y, batch_size=32, shuffle=False, *arrays):
    assert all(arrays[0].shape[0] == array.shape[0] for array in arrays)
    tensors = [torch.tensor(array, dtype=torch.long) for array in arrays]
    y_tensor = torch.tensor(y, dtype=torch.float32)
    tensors.append(y_tensor)

    ds = torch.utils.data.TensorDataset(*tensors)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_eval(args):
    set_seed(args.seed_num)
    train_x_left, train_x_entity, train_x_right, train_y = read_pickle(args.train_data_path)
    valid_x_left, valid_x_entity, valid_x_right, valid_y = read_pickle(args.valid_data_path)

    args.num_labels = train_y.shape[1]
    train_dataloader = get_dataloader(train_y, args.train_batch_size, True, train_x_left, train_x_entity, train_x_right)
    valid_dataloader = get_dataloader(valid_y, args.valid_batch_size, False, valid_x_left, valid_x_entity, valid_x_right)

    if args.model == 'bert':
        return None
        # bert_config = BertConfig(args.bert_config_path)
        # model = NERBert(bert_config, args)
        # model.load_state_dict(torch.load(args.bert_model_path), strict=False)
        # model = NERBert.from_pretrained('bert_chinese',
        #                                 # cache_dir='/home/dutir/yuetianchi/.pytorch_pretrained_bert',
        #                                 num_labels=args.num_labels)
    else:
        if args.embedding:
            word_embedding_matrix = read_pickle(args.embedding_data_path)
            args.vocab_size = len(word_embedding_matrix)
            model = AttentiveLSTM(args, word_embedding_matrix)
        else:
            logging.error("args.embedding should be true")
            return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.model == 'bert':
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if 'bert' not in n], 'lr': 5e-5, 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and ('bert' in n)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and ('bert' in n)],
             'weight_decay': 0.0}
        ]
        warmup_proportion = 0.1
        num_train_optimization_steps = int(
            train_samples / args.train_batch_size / args.gradient_accumulation_steps) * args.epochs

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    else:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    global_step = init_step
    best_score = 0.0

    logging.info('Start Training...')
    logging.info('init_step = %d' % global_step)
    for epoch_id in range(int(args.epochs)):

        train_loss = 0

        for step, train_batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in train_batch)
            train_x = (batch[0], batch[1], batch[2])
            train_y = batch[3]
            loss = model(train_x, train_y)
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            train_loss += loss.item()
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if args.do_valid and global_step % args.valid_step == 1:
                true_res = []
                pred_res = []
                valid_losses = []
                model.eval()
                for valid_step, valid_batch in enumerate(valid_dataloader):
                    valid_batch = tuple(t.to(device) for t in valid_batch)
                    valid_x = (valid_batch[0], valid_batch[1], valid_batch[2])
                    valid_y = valid_batch[3]
                    with torch.no_grad():
                        valid_logit = model(valid_x)
                    valid_loss = F.binary_cross_entropy_with_logits(valid_logit, valid_y)
                    valid_logit = F.sigmoid(valid_logit)
                    if args.model == 'bert':
                        # 第一个token是‘cls’
                        valid_losses.append(valid_loss.item())
                        true_res.extend(valid_y.detach().cpu().numpy())
                        pred_res.extend(valid_logit.detach().cpu().numpy())
                    else:
                        valid_losses.append(valid_loss.item())
                        true_res.extend(valid_y.detach().cpu().numpy())
                        pred_res.extend(valid_logit.detach().cpu().numpy())

                metric_res = acc_hook(pred_res, true_res)
                logging.info('Evaluation:step:{},train_loss:{},valid_loss:{},microf1:{},macrof1:{}'.
                             format(str(global_step), train_loss / args.valid_step, np.average(valid_losses),
                                    metric_res['loose_micro_f1'], metric_res['loose_macro_f1']))
                if metric_res['loose_micro_f1'] >= best_score:
                    best_score = metric_res['loose_micro_f1']
                    if args.model == 'bert':
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_dir = '{}_{}'.format('bert', str(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            output_config_file = os.path.join(output_dir, CONFIG_NAME)
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())
                    else:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': args.learning_rate,
                            'warm_up_steps': step
                        }
                        save_model(model, optimizer, save_variable_list, args)
                train_loss = 0.0


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    train_eval(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
