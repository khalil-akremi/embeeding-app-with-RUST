use candle_core::{Device, Tensor, DType, Result};
use candle_nn::{Linear, Module, VarBuilder, init};

// Simple embedding model
pub struct SimpleEmbedder {
    embedding_layer: Linear,
    device: Device,
}

impl SimpleEmbedder {
    pub fn new(vocab_size: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        let vs = VarBuilder::zeros(DType::F32, device);
        
        // Create embedding layer
        let embedding_layer = candle_nn::linear(vocab_size, embedding_dim, vs.pp("embed"))?;
        
        Ok(Self {
            embedding_layer,
            device: device.clone(),
        })
    }
    
    // Simple one-hot encoding
    fn one_hot(&self, token_ids: &[u32], vocab_size: usize) -> Result<Tensor> {
        let batch_size = token_ids.len();
        let mut one_hot = vec![0.0; batch_size * vocab_size];
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            if (token_id as usize) < vocab_size {
                one_hot[i * vocab_size + token_id as usize] = 1.0;
            }
        }
        
        Tensor::from_vec(one_hot, (batch_size, vocab_size), &self.device)
    }
    
    pub fn embed(&self, token_ids: &[u32]) -> Result<Tensor> {
        let vocab_size = self.embedding_layer.weight().dim(1)?;
        let one_hot = self.one_hot(token_ids, vocab_size)?;
        self.embedding_layer.forward(&one_hot)
    }
}