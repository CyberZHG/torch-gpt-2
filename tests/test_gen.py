from unittest import TestCase
import torch
import torch.nn as nn
from torch_gpt_2 import GPT_2, BytePairEncoding, generate, gelu


class TestGen(TestCase):

    def test_train_and_gen(self):
        token_dict = {chr(i): i for i in range(2 ** 9)}
        token_dict['Po'] = len(token_dict)
        token_dict['er'] = len(token_dict)
        net = GPT_2(
            n_vocab=len(token_dict),
            n_ctx=100,
            n_embd=30,
            n_head=5,
            n_layer=2,
            attention_activation=gelu,
            feed_forward_activation=gelu,
        )
        bpe = BytePairEncoding(token_dict=token_dict, bpe_rank={('P', 'o'): 0, ('e', 'r'): 1})
        texts = [
            'Power, give me more power!',
            'From the day forth, my arm changed.',
        ]
        space_encode = bpe.encode(' ')
        inputs = [bpe.encode(text) for text in texts]
        max_len = max(map(len, inputs))
        x = [encode + space_encode * (max_len - len(encode)) for encode in inputs]
        y = torch.LongTensor([encode[1:] + space_encode for encode in x])
        x = torch.LongTensor(x).repeat(1, 1)
        y = y.repeat(1, 1)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
        criterion = nn.CrossEntropyLoss()
        for i in range(100):
            optimizer.zero_grad()
            y_hat = net(x)
            loss = criterion(y_hat.view(-1, len(token_dict)), y.view(-1))
            loss.backward()
            optimizer.step()
            print('Round %3d : %.4f' % (i + 1, loss.item()), y_hat[0].argmax(dim=-1).tolist()[:10], y[0].tolist()[:10])

        texts = [
            'Power, give me more',
            'Power',
            'give me more ',
            'the day forth ',
            'From',
        ]
        results = generate(net, bpe, texts, length=30)
        # self.assertEqual(results[0][:len('Power, give me more power!')], 'Power, give me more power!')
