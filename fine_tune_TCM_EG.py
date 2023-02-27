
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import to_array
from keras.models import Model
from utils import load_data, data_generator, decoderTCM, metric_efficacy
from math import pow, floor
from keras.callbacks import LearningRateScheduler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

#  Configuration file path for the pre-trained language model
config_path = r'data/pretrain_model/bert_config.json'
checkpoint_path = r'data/pretrain_model/model.ckpt'
dict_path = r'data/pretrain_model/vocab.txt'

#  Path for saving the TCM-EG model weight file
save_path = r'output/checkpoint/formula2treat_model.weights'
date = '-cv-1'

# Model training parameter
maxlen = 60
batch_size = 4
epochs = 40

# Load the BERT vocabulary
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

# Generate BERT Tokenizer
tokenizer = Tokenizer(token_dict, do_lower_case=True)

# Load data set
trainData = load_data(r'data/dataset/train-1.txt')
evalData = load_data(r'data/dataset/eval-1.txt')
testData = load_data(r'data/dataset/eval-1.txt')


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

# build_transformer_model
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,
    ignore_invalid_weights=True  # Model without position embedding
)

# Optimizer configuration
adam = Adam(lr=1e-5)

# Getting the model output
output = maskCrossEntropyLoss(2)(model.inputs + model.outputs)
print('model.inputs---', model.inputs)
model = Model(model.inputs, output)
model.compile(optimizer=adam)  # Adam(1e-5)
model.summary()
outputEG = decoderTCM(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)
outputEG.set_model(model)


#  Get the loss of the model on each batch
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
history = LossHistory()

#  Model performance was evaluated at the end of each training round
class Evaluator(keras.callbacks.Callback):
    """
    Called at the end of each epoch
    """
    def __init__(self):
        super(Evaluator, self).__init__()

        self.results = {}
        self.embedding = {}
        self.jaccard_score = []

    def sim_efficacy(self, y_pred, y_true):
        sim_eff = metric_efficacy(y_pred, y_true)
        return sim_eff

    def on_epoch_end(self, epoch, logs=None):

        results = self.evaluate(evalData)  # 评测模型
        print(results)
        self.results[epoch] = results
        if (epoch+1) % 3 == 0:
            model.save_weights(save_path)  # 保存模型

        sim_scores = []
        for one_sample in results:
            y_true = one_sample['true']
            y_pred = one_sample['pred']
            sim_score = self.sim_efficacy(y_pred, y_true)[0]
            sim_scores.append(sim_score)
            # # 提取embedding
            # formula = one_sample['input']
            # # pred_label = results[0]['pred'][0]
            # # print(true_label, pred_label)
            # token_ids, seg_ids = tokenizer.encode(formula)
            # token_ids, seg_ids = to_array(token_ids, seg_ids)
            #
            # layer_model = Model(inputs=model.input, outputs=model.get_layer('MLM-Dense').output)
            # # embedding = layer_model.predict([token_ids, seg_ids])
            # embedding = list(
            #     layer_model.predict([token_ids, seg_ids]).astype(np.float64)[0].reshape(-1)
            # )
            # formula_embeddings.append({'input': formula, 'embedding': embedding})
            # self.embedding[epoch] = formula_embeddings
        mean_jaccard_scores = np.mean(sim_scores)
        print('sim_efficacy_score', mean_jaccard_scores)
        self.jaccard_score.append(mean_jaccard_scores)
        # self.results.append(results)

    def evaluate(self, data, search_num=1):
        records = []

        for formula, treatment in tqdm(data):
            treatment_ture = set(treatment.split('，'))
            treatment_ture = '，'.join([t for t in treatment_ture if t])
            treatment_pred = outputEG.generate(formula, maxlen, tokenizer, search_num)  #
            treatment_pred = set(treatment_pred.split('，'))
            treatment_pred = '，'.join([p for p in treatment_pred if p])
            records.append({'true': treatment_ture, 'pred': treatment_pred, 'input': formula})

        return records



if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(trainData, batch_size)
    train_generator.set_toknizer(tokenizer=tokenizer, max_length=maxlen)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator, history],
    )

    import json
    with open(r'output/log/F2T_train_log' + date +'.json', 'w', encoding='utf-8') as f:
        log_loss = {'train_loss': [str(l) for l in history.losses],
                    'preds': evaluator.results,
                    'embedding': evaluator.embedding,
                    'jaccard': evaluator.jaccard_score}
        f.write(json.dumps(log_loss))
