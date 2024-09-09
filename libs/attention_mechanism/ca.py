from libs.attention_mechanism.ba import BaseAttention

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
    
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  def test_ca(train_dataset):
    sample_ca = CrossAttention(num_heads=2, key_dim=512)
    for (x, y) in train_dataset.take(1):
        print(x[0].shape)
        print(x[1].shape)
        print(sample_ca(x[0], x[1]).shape)