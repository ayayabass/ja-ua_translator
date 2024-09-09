from libs.attention_mechanism.ba import BaseAttention

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
  def test_csa(train_dataset):
    sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
    for (x, y) in train_dataset.take(1):
        print(x[1].shape)
        print(sample_csa(x[1]).shape)