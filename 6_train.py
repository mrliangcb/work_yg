import os
import sys
import torch
from torch.nn.utils import clip_grad_norm_
import time
import glob
import copy

from model_5 import Model,run_model
from make_batch_4 import train_batch_out,Vocab
import config
from config import USE_CUDA, DEVICE
from adagrad_custom import AdagradCustom
# from utils import calc_running_avg_loss


print('是否cuda:',USE_CUDA)




def main(model_file_path=None):
    train_w2i_path = './word2id_dic.txt'
    train_i2w_path = './id2word_dic.txt'
    train_text_path = './fenci/txt_fenci.txt'
    train_label_path = './fenci/label_fenci.txt'
    
    vocab=Vocab(train_w2i_path,train_i2w_path)
    epoch=700
    best_loss=1000
    #建立模型
    net=Model()
    params = list(net.encoder.parameters()) + list(net.decoder.parameters()) + \
                 list(net.reduce_state.parameters())
    initial_lr = config.lr_coverage if config.is_coverage else config.lr
    optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
    ite=0
    # 二次训练
    
    if model_file_path is not None:
        state = torch.load(model_file_path)
        net.encoder.load_state_dict(state['encoder_state_dict'])
        net.decoder.load_state_dict(state['decoder_state_dict'])# , strict=False
        net.reduce_state.load_state_dict(state['reduce_state_dict'])
        old_ite=state['iter']
        ite=old_ite


    for epo in range(epoch):
        batch_gen=train_batch_out(train_text_path,train_label_path,vocab,config.batch_size) #yield实现
        start = time.time()
        for data in batch_gen:

            optimizer.zero_grad()
            input_data,output_data,ex_list=data
            # input_data,output_data=input_data.to(DEVICE),output_data.to(DEVICE)
            # for j in range(len(ex_list)):
            #     print(ex_list[j].article_oovs)
            loss=run_model(net,input_data,output_data)
            # print('是否cuda:',loss.is_cuda)
            if torch.isnan(loss):
                print('loss为0，不训练')
                print('loss为0，不训练')
                print('loss为0，不训练')
                print('loss为0，不训练')
                print('loss为0，不训练')
                print('loss为0，不训练')
                print('loss为0，不训练,epo:{},ite:{}'.format(epo,ite)) #出现脏数据，导致训练出现nan
                print(ex_list[0].src_article)
                print(ex_list[0].src_abs)
                with open(r'D:\MY_PGN\weight\reload\huai_data.txt', 'a',encoding='utf-8') as f_a:
                    f_a.write(' '.join(ex_list[0].src_article) + '\n')
                    f_a.write(' '.join(ex_list[0].src_abs) + '\n')
                    net=last_net #退回上一个版本
                    params = list(net.encoder.parameters()) + list(net.decoder.parameters()) + \
                        list(net.reduce_state.parameters())
                    optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

                continue
            last_net=copy.deepcopy(net)
            loss.backward()
            norm = clip_grad_norm_(net.encoder.parameters(), config.max_grad_norm)
            clip_grad_norm_(net.decoder.parameters(), config.max_grad_norm)
            clip_grad_norm_(net.reduce_state.parameters(), config.max_grad_norm)
            optimizer.step()
            # if ite%1==0:
            #     print('epo: %d ,steps %d, seconds for %.2f , loss: %.2f' % (epo,ite,time.time() - start, loss))
            if ite % 100 == 0:
                # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('epo: %d ,steps %d, seconds for %.2f , loss: %.2f' % (epo,ite,time.time() - start, loss))
                start = time.time()
                # if loss.item()<best_loss:
                #     best_loss=loss.item()
            if ite % 1000 == 0:
                net.save_model(epo,loss.item(), ite,optimizer)
            ite+=1


reload_path_glob=None
# weight_ROOT = "./weight"
# reload_path=os.path.join(weight_ROOT, 'reload','*.ckpt')
# reload_path_glob=glob.glob(reload_path)[0]
main(reload_path_glob)




























