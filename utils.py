from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import sequence_padding, open
import numpy as np

def load_data(full_file_path):
    D = []
    with open(full_file_path, encoding='utf-8') as f:  # utf-8, gbk
        for l in f:
            # print(l)
            med, dispat = l.strip().split('\t')
            D.append((med, dispat))
    return D

def load_data2(full_file_path):
    D = []
    with open(full_file_path, encoding='gbk') as f:  # utf-8, gbk
        for l in f:
            print(l)
            med, dispat = l.strip().split('\t')
            D.append((med, dispat))
    return D

class data_generator(DataGenerator):
    """
    Inherit from DataGenerator class, the Input model should have two parts: input and Label.
    Here the Label of the Input model is None, which will be captured from the input section.
    """
    def set_toknizer(self, tokenizer, max_length):
        self.tokenizer=tokenizer
        self.max_length = max_length

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (med, dispat) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(
                med, dispat, maxlen=self.max_length
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None  # Input:[batch_token_ids, batch_segment_ids] Label:None
                batch_token_ids, batch_segment_ids = [], []


class decoderTCM(AutoRegressiveDecoder):
    """
    Inherit AutoRegressiveDecoder, which is the logist to decode and output the treatment
    """
    def set_model(self, model):
        self.model = model

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(self.model).predict([token_ids, segment_ids])

    def generate(self, text, maxlen, tokenizer, beam_search=1):  # 适配于异病同治
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=beam_search)

        return tokenizer.decode(output_ids)


def metric_efficacy(pred, true):
    def eff_split(items, step=2):
        items = items.split('，')
        split_items = []
        for item in items:
            cut_num = int(len(item) / step)
            split_item = [item[2 * i: 2 * (i + 1)] for i in range(cut_num)]
            split_items.extend(split_item)
        # print(split_items)
        return split_items

    def eff_sim(eff_pred, eff_true):
        overlap = set(eff_pred) & set(eff_true)
        whole = set(eff_true) | set(eff_pred)
        reference = set(eff_true)
        sim1 = len(overlap) / len(reference)

        eff_pred_outer = set(eff_pred) - overlap
        eff_true_outer = set(eff_true) - overlap

        item_overlap_1 = [pred for pred in eff_pred_outer if pred[0] in [i[0] for i in eff_true_outer]]
        item_overlap_2 = [pred for pred in eff_pred_outer if pred[1] in [i[1] for i in eff_true_outer]]
        sim2 = (len(item_overlap_1)+len(item_overlap_2))/len(reference)

        # Lower bound on the SC score
        SC_score = sim1+sim2
        # The upper bound of the SD score
        SA_score = (len(whole)-(len(overlap)+2*(len(item_overlap_1)+len(item_overlap_2)))) / len(whole)
        # The upper bound of the SA score
        SD_score = 1 - SC_score

        return [SC_score, SA_score, SD_score]

    p, t = eff_split(pred), eff_split(true)
    sim = eff_sim(p, t)

    return sim