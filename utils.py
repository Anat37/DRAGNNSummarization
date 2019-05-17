from data import *
from nltk.translate.bleu_score import sentence_bleu
import math
import time
from decoder import *
from pointerTBRU import BeamSearchProviderTBRU, LSTMTBRU, AttentionTBRU, ConcatTBRU, PointerTBRU, ContextTBRU, CoverageAttentionTBRU, ConcatStateTBRU

def clean_sentence(target):
    if not isinstance(target, list):
        target = target.tolist()
    if STOP_DECODING_ID in target:
        target = target[:target.index(STOP_DECODING_ID)]
    while PAD_TOKEN_ID in target:
        target.remove(PAD_TOKEN_ID)
    return target

def calculate_bleu(result, target, weights): #TODO
    result = clean_sentence(result)
    target = clean_sentence(target)
    BLEUscore = sentence_bleu([target], result, weights=weights)
    return BLEUscore
    
def calculate_bleu_ngramm(result, target, n):
    cnt = 0
    for i in range(len(result) - n + 1):
        if check_ngramm_in_string(result[i:i+n], target):
            cnt += 1
    return cnt / (len(result) - n + 1)
    
def check_ngramm_in_string(ngramm, target):
    for i in range(len(target) - len(ngramm) + 1):
        flag = True
        for j in range(len(ngramm)):
            flag = flag and (ngramm[j] == target[i + j])
        if flag:
            return True
    return False

def calculate_logits_bleu_and_rouge(logits, target, weights):
    result = logits.argmax(-1).cpu().detach().numpy().T
    
    target = target.T
    #print(result)
    #print(target)
    bleu = 0.
    rouge = 0.
    for i in range(result.shape[0]):
        bleu += calculate_bleu(result[i], target[i], weights)
        rouge += calculate_bleu(target[i], result[i], weights)
    return bleu / result.shape[0], rouge / result.shape[0]

def generate_summary(article, model, beam_width):
    symbols = [START_DECODING_ID]
    beam_ids = [0]
    probs = [1.]
    result = np.array([[]*beam_width])
    X_batch = LongTensor(article)
    inputs = InputLayerState("input", False, X_batch)
    model.eval_run_encoder(inputs)
    for k in range(120):
        hidden = model.decode((LongTensor([symbols]), LongTensor(beam_ids)))
        new_probs = []
        new_result = []
        for i, s in enumerate(symbols):
            values, indices = hidden[i].topk(beam_width)
            new_probs.extend((values + probs[i]).cpu().detach().tolist())
            for j in range(beam_width):
                tmp = result[i].tolist()
                tmp.append(indices[j].item())
                new_result.append(tmp)
        top_idx = np.argsort(new_probs)[-beam_width:]
        probs = np.array(new_probs)[top_idx]
        result = np.array(new_result)[top_idx]
        symbols = result[:,-1]
        if symbols[0] == STOP_DECODING_ID:
            break
        beam_ids = top_idx // beam_width
        #symbol[0] == STOP_DECODING_ID
    return result[0]

def gen_and_print_summary(batcher, model, beam_width):
    article_text, target, decoder_input, unk_words = batcher.get_random_sample()
    result = generate_summary(article_text, model, beam_width)
    result = outputids2words(result, vocab, unk_words)
    target = outputids2words(target, vocab, unk_words)
    print('result is \n' + result)
    print('target is \n' + target)
    print('BLEU = {:.4f}'.format(sentence_bleu([target], result, weights=(0.33, 0.33, 0.33, 0))))
    
def gen_and_print_summary_by_target(batcher, model):
    article_text, target, decoder_inputs, unk_words = batcher.get_random_sample()
    X_batch, y_batch, decoder_batch = LongTensor(article_text), LongTensor(target), LongTensor(decoder_inputs)
    inputs = InputLayerState("input", True, X_batch)
    targetLayer = InputLayerState("target", True, decoder_batch)
    logits = model.train_run(inputs, targetLayer)
    

    print(calculate_logits_bleu_and_rouge(logits, np.array([target]).T, (0.33,0.33,0.33,0)))
    result = logits.argmax(-1).squeeze(0).squeeze(-1).cpu().detach().tolist()
    
    result = outputids2words(result, vocab, unk_words)
    target = outputids2words(target, vocab, unk_words)
    print('result is \n' + result)
    print('target is \n' + target)

def get_logits(model, X_batch, decoder_batch):
    inputs = InputLayerState("input", True, X_batch)
    targetLayer = InputLayerState("target", True, decoder_batch)
    logits = model.train_run(inputs, targetLayer)
    return logits
    
def get_logits2(model, X_batch, decoder_batch):
    inputs = InputLayerState("input", True, X_batch)
    batch_size = X_batch.shape[1]
    symbols = LongTensor([[START_DECODING_ID]*batch_size])
    result = []
    model.eval_run_encoder(inputs, beam_search=False)
    for k in range(decoder_batch.shape[0]):
        hidden = model.decode(symbols)
        symbols = hidden.argmax(-1)
        result.append(hidden)
    result = torch.stack(result)
    result.requires_grad_(True)
    return result

def calc_loss(logits, target, mask):
    #print(target.shape)
    #print(logits.shape)
    probs = torch.gather(logits, 2, target.unsqueeze(-1))
    loss = -torch.log(probs.squeeze(-1))*mask
    loss = loss.sum(0)
    return loss.mean()

def do_epoch(model, criterion, data, batch_size, bleu_weights, optimizer=None, cov_loss=False, mask_layer=None):  
    epoch_loss = 0.
    sum_cov_loss = 0.
    bleu = 0.
    rouge = 0.
    batch_cnt = 1
    is_train = not optimizer is None
    model.train(is_train)

    with torch.autograd.set_grad_enabled(is_train):
        for j, (article_text, target, encoder_mask, target_mask, decoder_inputs) in enumerate(data.generator()):
            batch_cnt =  j + 1
            X_batch, y_batch, decoder_batch, target_mask = LongTensor(article_text), LongTensor(target), LongTensor(decoder_inputs), FloatTensor(target_mask)
            if mask_layer is not None:
                mask_layer.reset()
                encoder_mask = FloatTensor(encoder_mask)
                mask_layer.hiddens = encoder_mask.unsqueeze(-1)
            
            if is_train:
                logits = get_logits(model, X_batch, decoder_batch)
            else:
                logits = get_logits2(model, X_batch, decoder_batch)
            #print(logits.view(1, -1).shape)
            #loss = criterion(logits.view(-1, logits.shape[-1]), y_batch.view(-1))
            loss = calc_loss(logits, y_batch, target_mask)
            epoch_loss += loss.item()
            #print(epoch_loss)
            coverage_loss = torch.tensor(0., requires_grad=True).cuda()
            
            cur_bleu, cur_rouge = calculate_logits_bleu_and_rouge(logits, target, bleu_weights)
            
            bleu += cur_bleu
            rouge += cur_rouge
            
            if cov_loss:
                norm_fact = y_batch.shape[0] * X_batch.shape[1]
                coverage = model.net.get_layer('coverage_layer').get_all()
                attention = model.net.get_layer('attention_layer').get_all()
                for i in range(len(attention)):
                    coverage_loss = coverage_loss + ((torch.min(attention[i], coverage[i]) * target_mask[i]).sum() / norm_fact)
                sum_cov_loss += coverage_loss.item()
                #print(coverage_loss._grad)
                loss += 0.2 * coverage_loss
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
            
            print('\r[{}]: Loss = {:.4f}, Cov_Loss = {:.4f}, BLEU = {:.4f}, ROUGE = {:.4f}'.format(j, loss.item(), coverage_loss.item(),  cur_bleu, cur_rouge), end='')
    
    
    #gen_and_print_summary_by_target(data, model)
    return epoch_loss, sum_cov_loss, bleu / batch_cnt, rouge / batch_cnt

def fit(model, criterion, optimizer, train_data, epochs_count=1, 
        batch_size=16, val_data=None, val_batch_size=None, cov_loss=False, mask_layer=None):
    if not val_data is None and val_batch_size is None:
        val_batch_size = batch_size
        
    bleu_weights = (0.5, 0.5, 0, 0)
    for epoch in range(epochs_count):
        start_time = time.time()
        train_loss, sum_cov_loss, bleu, rouge = do_epoch(model, criterion, train_data, batch_size, bleu_weights, optimizer, cov_loss, mask_layer)
        output_info = '\rEpoch {} / {}, Epoch Time = {:.2f}s: Train Loss = {:.4f}: Cov_Loss = {:.4f}, BLEU = {:.4f}, ROUGE = {:.4f}'
        if not val_data is None:
            val_loss, sum_cov_loss, bleu, rouge = do_epoch(model, criterion, val_data, val_batch_size, bleu_weights, None, False, mask_layer)
            epoch_time = time.time() - start_time
            output_info += ', Val Loss = {:.4f}'
            print(output_info.format(epoch+1, epochs_count, epoch_time, train_loss, sum_cov_loss, bleu, rouge, val_loss))
        else:
            epoch_time = time.time() - start_time
            print(output_info.format(epoch+1, epochs_count, epoch_time, train_loss, sum_cov_loss, bleu, rouge))
    print()
    #gen_and_print_summary(train_data, model, 10)