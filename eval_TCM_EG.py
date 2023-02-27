from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import to_array
from keras.models import Model
from utils import load_data, data_generator, decoderTCM, metric_efficacy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd

# Configuration file path for the pre-trained language model
config_path = r'data/pretrain_model/bert_config.json'
checkpoint_path = r'data/pretrain_model/bert_model.ckpt'
dict_path = r'data/pretrain_model/vocab.txt'
weight_path = r'output/checkpoint/formula2treat.weights'
# Model parameter
maxlen = 60

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class maskCrossEntropyLoss(Loss):
    """
    Custom set the LOSS function of the model: This first inherits the loss function,
    and then uses the MASK mechanism to calculate the cross-entropy
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]
        y_mask = y_mask[:, 1:]
        y_pred = y_pred[:, :-1]
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,
)


output = maskCrossEntropyLoss(2)(model.inputs + model.outputs)
print('model.inputs---', model.inputs)
model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


outputEG = decoderTCM(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)
outputEG.set_model(model)

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.results = {}
        self.embedding = {}
        self.pred4emb = []

    def evaluate(self, data, search_num=1):
        # print(data)
        records = []
        for formula, treatment in data:

            treatment_ture = set(treatment.split('，'))
            treatment_ture = '，'.join([t for t in treatment_ture if t])
            treatment_pred = outputEG.generate(formula, maxlen, tokenizer, search_num)  #
            treatment_pred = set(treatment_pred.split('，'))
            treatment_pred = [p for p in treatment_pred if p]
            records.append({'true': treatment_ture, 'pred': treatment_pred, 'input': formula})

        self.results[0] = records

        get_emb = True
        if get_emb:
           # Get embedding
            layer_model = Model(inputs=model.input, outputs=model.get_layer('MLM-Activation').output)
            # print(results)
            embeddings = []
            for index, result in enumerate(records):
                true_label = result['true']
                pred_label = result['pred']
                pred_label = '，'.join([p for p in pred_label if p])
                sim_score = metric_efficacy(pred_label, true_label)

                inputs = result['input']
                print(true_label, pred_label, sim_score)
                input_token_ids, input_seg_ids = tokenizer.encode(inputs)
                input_token_ids, input_seg_ids = to_array(input_token_ids, input_seg_ids)

                input_label_embedding = layer_model.predict([input_token_ids, input_seg_ids]).astype(np.float64)
                input_label_embedding = np.mean(input_label_embedding, axis=0).reshape(-1).tolist()
                # print(input_label_embedding)
                # pred_token_ids, pred_seg_ids = tokenizer.encode(pred_label)
                # pred_token_ids, pred_seg_ids = to_array(pred_token_ids, pred_seg_ids)
                # pred_label_embedding = list(
                #     layer_model.predict([pred_token_ids, pred_seg_ids]).astype(np.float64)[0].reshape(-1))
                self.pred4emb.append([inputs, true_label, pred_label, sim_score])

                emb = {'input': true_label, 'embedding': input_label_embedding}  # , pred_label: pred_label_embedding
                embeddings.append(emb)
                # print(emb)
            self.embedding['0'] = embeddings


if __name__ == '__main__':
    # Loading the test dataset
    testData = load_data(r'data/dataset/eval-1.txt')
    evaluator = Evaluator()
    model.load_weights(weight_path)
    evaluator.evaluate(testData)

    import json
    with open(r'output/log/get_F2T_eval.json', 'w', encoding='utf-8') as f:
        log = {
                    'preds': evaluator.results,
                    'embedding': evaluator.embedding
               }
        f.write(json.dumps(log))
    df = pd.DataFrame(evaluator.pred4emb).to_excel(r'output/log/get_treatment_emb_from_T2D.xls')
