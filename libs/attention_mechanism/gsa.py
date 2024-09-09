from libs.attention_mechanism.ba import BaseAttention

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
  def test_gsa(train_dataset):
    sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
    for (x, y) in train_dataset.take(1):
        print(x[0].shape)
        print(sample_gsa(x[0]).shape)