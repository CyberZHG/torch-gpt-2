from unittest import TestCase
from torch_gpt_2 import GPT_2


class TestNet(TestCase):

    def test_init(self):
        net = GPT_2(
            n_vocab=327,
            n_ctx=100,
            n_embd=16,
            n_head=4,
            n_layer=4
        )
        print(net)
