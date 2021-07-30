import sys

sys.path.append("../Skip_Gram_NGE/")
from skip_gram_nge_model import SkipGramModel
from input_data import InputData
import torch.optim as optim
from tqdm import tqdm
import torch
import argumentparser as argumentparser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

args = argumentparser.ArgumentParser()
WINDOW_SIZE = args.window_size  # 上下文窗口c
BATCH_SIZE = args.batch_size  # mini-batch
MIN_COUNT = args.min_count  # 需要剔除的 低频词 的频
EMB_DIMENSION = args.embed_dimension  # embedding维度
LR = args.learning_rate  # 学习率
NEG_COUNT = args.neg_count  # 负采样数


class Word2Vec:
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = InputData(input_file_name, MIN_COUNT)
        self.model = SkipGramModel(self.data.word_count, EMB_DIMENSION).to(device)
        self.lr = LR
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        # self.model.load_state_dict(torch.load("../results/skipgram_nge.pkl"))
        print("SkipGram Training......")
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count", pairs_count)
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(5 * batch_count)))
        for _ in process_bar:
            pos_pairs = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
            pos_w = [int(pair[0]) for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs]
            neg_v = self.data.get_negative_sampling(pos_pairs, NEG_COUNT)
            pos_w = pos_w
            pos_v = pos_v
            neg_v = neg_v

            self.optimizer.zero_grad()      # 把模型中参数的梯度设为0
            loss = self.model.forward(pos_w, pos_v, neg_v)

            """
            那么为什么optimizer.step()需要放在每一个batch训练中，而不是epoch训练中？
            这是因为现在的mini-batch训练模式是假定每一个训练集就只有mini-batch这样大，因此实际上可以将每一次mini-batch看做是一次训练，一次训练更新一次参数空间，因而optimizer.step()放在这里。
            Epoch由一个或多个Batch组成
            """
            loss.backward()
            self.optimizer.step()

            # 进度条设置
            process_bar.set_postfix(loss=loss.data)
            process_bar.update()
        torch.save(self.model.state_dict(), "../results/skipgram_nge_0730_gpu_windows.pkl")
        self.model.save_embedding(self.data.id2word_dict, self.output_file_name)


if __name__ == '__main__':
    import time

    t_start = time.time()
    w2v = Word2Vec(input_file_name='../data/text8.txt',
                   output_file_name="../results/skip_gram_neg_0730_gpu_windows.txt")
    w2v.train()
    t_end = time.time()
    print("训练消耗时间：{}".format(t_end - t_start))
