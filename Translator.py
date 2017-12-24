import os
import math
import pandas as pd
import seaborn as sns
import torch
from torch.autograd import Variable

class Translator(object):
    """Class to generate translations from beam search and to plot attention heatmap.
    
    Args:
        model:          pre-trained translation model with encode and decode methods
        bpe_model_path: path to trained sentencepiece model from byte pair encoding
        beam_size:      number of beams for beam search
        alpha:          low alpha means high penalty on hypothesis length
        beta:           low beta means low penalty on hypothesis coverage
        max_len:        max length for hypothesis
              
    """
    def __init__(self, model, bpe_model_path, beam_size=64, alpha=0.1, beta=0.3, max_len=64):
        self.model = model
        self.bpe_size = model.bpe_size
        self.beam_size = beam_size
        self.alpha = alpha
        self.beta = beta
        self.max_len = max_len
        self.bpe_model_path = bpe_model_path
        self.encode_cmd = "spm_encode --model=" + bpe_model_path + " --output_format=id"
        self.decode_cmd = "spm_decode --model=" + bpe_model_path + " --input_format=id"
        self.model.eval()
        
    def word2id(self, word_input):
        word_input = word_input.replace('"', r'\"')
        encode_cmd = 'echo "' + word_input + '" |' + self.encode_cmd
        id_output = os.popen(encode_cmd).read()[:-1] # drop the \n at the end
        id_output = [int(id) for id in id_output.split()]
        return id_output
    
    def id2word(self, id_input):
        if id_input == "2":
            return "</s>"
        decode_cmd = "echo '" + id_input + "' |" + self.decode_cmd
        word_output = os.popen(decode_cmd).read()[:-1] # drop the \n at the end
        return word_output
     
    def translate(self, source):
        logLikelihoods = []
        preds = []
        atten_probs = []
        coverage_penalties = []
        beam_size = self.beam_size
        remaining_beams = self.beam_size
        EOS_id = 2
        
        # generate context from source
        src_id = self.word2id(source)
        src_pieces = [self.id2word(str(i)) for i in src_id]
        src_id = Variable(torch.LongTensor(src_id).unsqueeze(0).cuda(), volatile=True)
        context, mask_src = self.model.encode(src_id)
        
        # predict the first word
        decode_input = Variable(torch.LongTensor([1]).unsqueeze(1).cuda())
        out, coverage = self.model.decode(decode_input, context, mask_src)
        scores, scores_id = out.view(-1).topk(beam_size)
        beam_index = scores_id / self.bpe_size
        pred_id = (scores_id - beam_index*self.bpe_size).view(beam_size, -1)
        decode_input = torch.cat((decode_input.repeat(beam_size ,1), pred_id), 1)
        context = context.repeat(beam_size, 1, 1)
        
        # continus to predict next work until <EOS>
        step = 1
        while step <= self.max_len and remaining_beams>0:
            step += 1
            out, coverage = self.model.decode(decode_input, context, mask_src) 
            out = out.view(remaining_beams, -1, self.bpe_size)
            out = scores.unsqueeze(1) + out[:, -1, :]
            scores, scores_id = out.view(-1).topk(remaining_beams)
            beam_id = scores_id / self.bpe_size
            pred_id = (scores_id - beam_id*self.bpe_size).view(remaining_beams, -1)
            decode_input = torch.cat((decode_input[beam_id], pred_id), 1) 
            # remove finished beams
            finished_index = decode_input[:, -1].eq(EOS_id).data.nonzero().squeeze()
            continue_index = decode_input[:, -1].ne(EOS_id).data.nonzero().squeeze()
            for idx in finished_index:
                logLikelihoods.append(scores[idx].data[0])
                preds.append(decode_input[idx,:].data.tolist())
                atten_prob = torch.sum(coverage[idx,:,:], dim=0)
                atten_probs.append(coverage[idx,:,:])
                coverage_penalty = torch.log(atten_prob.masked_select(atten_prob.le(1)))
                coverage_penalty = self.beta * torch.sum(coverage_penalty).data[0]
                coverage_penalties.append(coverage_penalty)       
                remaining_beams -= 1
            if len(continue_index) > 0:
                scores = scores.index_select(0, continue_index)
                decode_input = decode_input.index_select(0, continue_index)
                context = context.index_select(0, continue_index)    
        
        # normalize the final scores by length and coverage 
        len_penalties = [math.pow(len(pred), self.alpha) for pred in preds]
        final_scores = [logLikelihoods[i]/len_penalties[i] + coverage_penalties[i] for i in range(len(preds))]
        sorted_scores_arg = sorted(range(len(preds)), key=lambda i:-final_scores[i])
        best_beam = sorted_scores_arg[0]
        tgt_id = ' '.join(map(str, preds[best_beam]))
        target = self.id2word(tgt_id)
        tgt_pieces = [self.id2word(str(i)) for i in preds[best_beam]]
        attn = [atten_probs[best_beam], src_pieces, tgt_pieces]
        return target, attn 
    
    def attention_heatmap(self, attn):
        """Draw attention heatmap using attn returned by translate()."""
        df = pd.DataFrame(attn[0].data.cpu().numpy(), index=attn[2][1:], columns=attn[1])
        sns.heatmap(df, linewidths=.005, xticklabels=1, yticklabels=1)   