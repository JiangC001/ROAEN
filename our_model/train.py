'''
Description:
version:
Author: J Chen
Date: 2023-08-09 12:02:32
'''
import os
import sys
import copy
import random
import logging
import argparse
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from memory_profiler import profile

from models.ourmodel_z_d import OurModelBertClassifier, Myloss, init_weights

from data_utils_ import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp
from base_module import PrintColor
from matplotlib import pyplot as plt

import pyttsx3 #
engine = pyttsx3.init()
engine.setProperty("voice","HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0")
engine.setProperty('rate', 150)

log = PrintColor.Log()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''

    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            # need download
            tokenizer = Tokenizer4BertGCN(opt.max_length, 'D:/bert-base-uncased/')
            bert = BertModel.from_pretrained('D:/bert-base-uncased/')
            self.model = opt.model_class(bert, opt).to(opt.device)
            if self.opt.init_weight:
                self.model.apply(init_weights)
            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)

        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_length=opt.max_length,
                data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))

            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab,
                embed_dim=opt.embed_dim,
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')

            logger.info(
                "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab),
                                                                                                      len(post_vocab),
                                                                                                      len(pos_vocab),
                                                                                                      len(dep_vocab),
                                                                                                      len(pol_vocab)))

            # opt.tok_size = len(token_vocab)
            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)  # pos

            vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

        # batch
        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer
    # train
    @profile(precision=4, stream=open('test.log', 'w+'))
    def _train(self, criterion, my_criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        train_loss_list, train_acc_list, test_acc_list, test_f1_list = [], [], [], []

        my_loss_sta_ave_tensor = torch.zeros(self.opt.polarities_dim,self.opt.attention_dim*2).numpy()

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch+1))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                # print(my_loss_sta_ave_tensor)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs,penal = self.model(inputs)

                my_loss_sta_ave_tensor = (my_loss_sta_ave_tensor + outputs[-1].detach().cpu().numpy())/2

                targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.losstype is not None:
                    loss1 = criterion(outputs[0], targets) + penal
                    loss2 = criterion(outputs[1], targets) + penal

                    loss3 = my_criterion(outputs[2], outputs[3], torch.Tensor(my_loss_sta_ave_tensor))

                else:
                    loss1 = criterion(outputs[0], targets)
                    loss2 = criterion(outputs[1], targets)
                    loss3 = my_criterion(outputs[2], outputs[3], torch.Tensor(my_loss_sta_ave_tensor))

                '''lambada*bgcn+attention'''
                loss = self.opt.lambda_1*loss1 + loss2 + self.opt.lambda_2*loss3
                loss.requires_grad_()

                print(log.blue(f'epoch{epoch+1}'),': ',log.green(f'{i_batch+1}'),'bach bgcn_loss(*lambda_1) is:',log.red(f'{loss1*self.opt.lambda_1}'),'gate_loss为',\
                      log.red(f'{loss2}'),'my_loss(*lambda_2) is:',log.red(f'{loss3*self.opt.lambda_2}'),'all Loss is:',log.magenta(f'{loss}'))
                # print(inputs[0].size())

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:

                    n_correct += (torch.argmax(outputs[1], -1) == targets).sum().item()
                    n_total += len(outputs[1])
                    train_acc = n_correct / n_total

                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc

                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name,
                                                                                          self.opt.dataset, test_acc,
                                                                                          f1)
                            if self.opt.dataset == 'restaurant':
                                if test_acc > self.opt.restaurant_acc:
                                    engine.say(f"max acc {round(100*test_acc,3)}")
                                    engine.runAndWait()
                            elif self.opt.dataset == 'laptop':
                                if test_acc > self.opt.laptop_acc:
                                    engine.say(f"max acc {round(100*test_acc,3)}")
                                    engine.runAndWait()

                            elif self.opt.dataset == 'twitter':
                                if test_acc > self.opt.twitter_acc:
                                    engine.say(f"max acc {round(100*test_acc,3)}")
                                    engine.runAndWait()

                            elif self.opt.dataset == 'restaurant15':
                                if test_acc > self.opt.restaurant15_acc:
                                    engine.say(f"max acc {round(100*test_acc,3)}")
                                    engine.runAndWait()

                            elif self.opt.dataset == 'restaurant16':
                                if test_acc > self.opt.restaurant16_acc:
                                    engine.say(f"max acc {round(100*test_acc,3)}")
                                    engine.runAndWait()

                            self.best_model = copy.deepcopy(self.model)  #

                            logger.info(log.cyan('>> saved: {}'.format(model_path)))


                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info(log.yellow(
                        '  {}   loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(global_step / self.opt.log_step,loss.item(), train_acc,
                                                                                         test_acc, f1)))
                    train_loss_list.append(loss.item())
                    train_acc_list.append(train_acc)
                    test_acc_list.append(test_acc)
                    test_f1_list.append(f1)

        result_dict = {'train_loss_list': train_loss_list,
                       'train_acc': train_acc_list,
                       'test_acc_list': test_acc_list,
                       'test_f1_list': test_f1_list}

        return max_test_acc, max_f1, model_path, result_dict

    def _evaluate(self, show_results=False):
        with torch.no_grad():
            # switch model to evaluation mode
            self.model.eval()
            n_test_correct, n_test_total = 0, 0
            targets_all, outputs_all = None, None

            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, penal = self.model(inputs)
                n_test_correct += (torch.argmax(outputs[1], -1) == targets).sum().item()
                n_test_total += len(outputs[1])
                targets_all = torch.cat((targets_all, targets),
                                        dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs[1]),
                                        dim=0) if outputs_all is not None else outputs[1]
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()

        if show_results:
            report = metrics.classification_report(labels, predic,digits=4)  # precision、recall、accuracy、f1-score，digits
            confusion = metrics.confusion_matrix(labels, predic)
            print(confusion)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        criterion = nn.CrossEntropyLoss()

        my_criterion = Myloss(self.opt).to(self.opt.device)

        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        if 'bert' not in self.opt.model_name:
            self._reset_params()

        max_test_acc, max_f1, model_path, result_dict = self._train(criterion, my_criterion, optimizer,
                                                                    max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        '''state = {‘net’:model.state_dict(), ‘optimizer’:optimizer.state_dict(), ‘epoch’:epoch}'''

        if self.opt.dataset == 'laptop':
            if max_test_acc_overall >= self.opt.save_laptop_acc:
                torch.save(self.best_model.state_dict(), model_path)
        elif self.opt.dataset == 'restaurant':
            if max_test_acc_overall >= self.opt.save_restaurant_acc:
                torch.save(self.best_model.state_dict(), model_path)
        elif self.opt.dataset == 'twitter':
            if max_test_acc_overall >= self.opt.save_twitter_acc:
                torch.save(self.best_model.state_dict(), model_path)
        elif self.opt.dataset == 'restaurant15':
            if max_test_acc_overall >= self.opt.save_restaurant15_acc:
                torch.save(self.best_model.state_dict(), model_path)
        elif self.opt.dataset == 'restaurant16':
            if max_test_acc_overall >= self.opt.save_restaurant16_acc:
                torch.save(self.best_model.state_dict(), model_path)

        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info(log.blue(f'max_test_acc_overall:{max_test_acc_overall}'))
        logger.info(log.blue(f'max_f1_overall:{max_f1_overall}'))

        self._test()

        return result_dict ,max_test_acc_overall, max_f1_overall

def _matplotlib(x, *args, **kwargs):
    loss, train_acc, test_acc, f1 = x.values()
    plt.figure()
    plt.plot(loss, color='blue', label='loss')
    plt.legend()
    plt.xlabel('each step(5_batch)')
    plt.ylabel('train_loss')
    plt.show()

    plt.figure()
    plt.plot(train_acc, color='red', label='train_acc')
    plt.plot(test_acc, color='blue', label='test_acc')
    plt.plot(f1, color='cyan', label='test_f1')
    plt.legend()
    plt.xlabel('each step(5_batch)')
    plt.ylabel('0-1')
    plt.show()


def main():

    dataset_list = ['dataset','dataset_']
    dataset = dataset_list[1]
    model_classes = {
        'ourmodel_bert': OurModelBertClassifier,
    }
    dataset_files = {
        'restaurant': {
            'train': rf'./{dataset}/Restaurants_corenlp/train_write.json',
            'test': rf'./{dataset}/Restaurants_corenlp/test_write.json',
        },
        'laptop': {
            'train': rf'./{dataset}/Laptops_corenlp/train_write.json',
            'test': rf'./{dataset}/Laptops_corenlp/test_write.json'
        },
        'twitter': {
            'train': rf'./{dataset}/Tweets_corenlp/train_write.json',
            'test': rf'./{dataset}/Tweets_corenlp/test_write.json',
        },
        'restaurant15': {
            'train': rf'./{dataset}/Restaurants15_corenlp/train_write.json',
            'test': rf'./{dataset}/Restaurants15_corenlp/test_write.json',
        },
        'restaurant16': {
            'train': rf'./{dataset}/Restaurants16_corenlp/train_write.json',
            'test': rf'./{dataset}/Restaurants16_corenlp/test_write.json',
        }
    }

    input_colses = {

        'ourmodel_glove': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'short_mask'],
        'ourmodel_bert': ['adj_f', 'adj_b', 'adj_f_aspect', 'adj_b_aspect', 'text_bert_indices', 'bert_segments_ids', \
                       'attention_mask','text_len', 'post_1', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask', 'polarity']}

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ourmodel_bert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='twitter', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str,
                        help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)  # l2
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)  # batch
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')  # droupout
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')  # GCN droupout
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--attention_heads', default=5, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./dataset_/Laptops_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)  #

    parser.add_argument('--parseadj', default=False, action='store_true',help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default=None, type=str,help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")

    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)  # bert
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')  # bert dropout
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)

    parser.add_argument('--RGCNlayer_num',default = 2,help = 'RGCN layer')
    parser.add_argument('--ARGCNlayer_num',default = 2,help = 'aspectRGCN layer')
    parser.add_argument('--c_c_Attention_num',default = 3,help = 'context_context_attention_layer_num')
    parser.add_argument('--c_a_Attention_num',default = 3,help = 'context_aspect_attention_layer_num')
    # parser.add_argument('--gate_Attention_num',default = 1,help = 'gate_attention_layer_num')
    parser.add_argument('--c_c_heads', default=5, type=int, help='number of context-context multi-attention heads')
    parser.add_argument('--c_a_heads', default=5, type=int, help='number of context-aspect multi-attention heads')
    # parser.add_argument('--gate_heads', default=1, type=int, help='number of gate_attention heads')

    parser.add_argument('--attention_dim', default=200, type=int, help='attention h dim')
    parser.add_argument('--bgcn_dim', default=200, type=int, help='attention h dim')
    parser.add_argument('--rand', default=0.01, help='rand')
    parser.add_argument('--lambda_1', default=0.39, help='lambda1 → bgcn loss')  # bgcn loss
    parser.add_argument('--lambda_2', default=0.000002, help='lambda2 → my_loss loss')  # my_loss
    parser.add_argument('--init_weight',default=False, help='init_wei-；ght')
    # save
    parser.add_argument('--save_laptop_acc',default=0.82, help='laptop_base_acc')
    parser.add_argument('--save_restaurant_acc',default=0.874, help='restaurant_base_acc')
    parser.add_argument('--save_twitter_acc',default=0.784, help='twitter_base_acc')
    parser.add_argument('--save_restaurant15_acc', default=0.856, help='restaurant15_base_acc')
    parser.add_argument('--save_restaurant16_acc', default=0.92, help='restaurant16_base_acc')
    #say
    parser.add_argument('--restaurant_acc',default=0.874, help='say_restaurant_acc')
    parser.add_argument('--laptop_acc',default=0.82, help='say_laptop_acc')
    parser.add_argument('--twitter_acc',default=0.784, help='say_twitter_acc')
    parser.add_argument('--restaurant15_acc',default=0.856, help='say_restaurant15_acc')
    parser.add_argument('--restaurant16_acc',default=0.92, help='say_restaurant16_acc')

    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print('bert' in opt.model_name)

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('log'):
        os.makedirs('log', mode=0o777)
    log_file = 'gcn{}-{}-{}-{}-{}-{}.log'.format(opt.lambda_1,opt.lambda_2,opt.bgcn_dim,opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H_%M_%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    ins = Instructor(opt)
    results_dict, acc, f1 = ins.run()

    return results_dict ,acc, f1

if __name__ == '__main__':

    start = time.time()
    logger.info(f'start run time:{time.asctime()}')
    result_dicct ,acc ,f1 = main()
    _matplotlib(result_dicct)
    stop = time.time()
    logger.info(f'stop run time:{time.asctime()}')

    ConvertedSec = str(datetime.timedelta(seconds=stop - start))
    logger.info(f'run time (hour:minute:second): {ConvertedSec}')

