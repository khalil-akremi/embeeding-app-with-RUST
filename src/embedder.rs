use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use crate::analyzers::Anchors;

pub struct MiniLMEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl MiniLMEmbedder {
    pub fn new() -> Result<Self> {
        println!("Loading all-MiniLM-L6-v2 model...");
        
        let device = Device::Cpu;
        let repo = Repo::with_revision(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            RepoType::Model,
            "main".to_string(),
        );
        
        let api = Api::new()?;
        let repo = api.repo(repo);
        
        // Download model files
        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;
        
        // Load config
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        
        // Load weights
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        
        println!("Model loaded successfully!");
        
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
    
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(E::msg)?;
        
        let token_ids = tokens.get_ids();
        let attention_mask = tokens.get_attention_mask();
        
        // Convert to tensors
        let token_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        
        // Forward pass
        let embeddings = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        
        // Mean pooling
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        
        // Normalize
        let embeddings = self.normalize_l2(&embeddings)?;
        
        // Convert to Vec<f32>
        let embeddings = embeddings.squeeze(0)?.to_vec1::<f32>()?;
        
        Ok(embeddings)
    }
    
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        
        for (i, text) in texts.iter().enumerate() {
            if i % 10 == 0 {
                println!("Processing chunk {}/{}", i + 1, texts.len());
            }
            let embedding = self.embed(text)?;
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }
    
    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
    
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot / (norm_a * norm_b)
    }
    
    pub fn create_anchors(&self) -> Result<Anchors> {
        println!("Creating anchor embeddings...");
        
        // Sentiment anchors
        let sentiment_anchors = Anchors::sentiment_anchors();
        let positive = self.embed(sentiment_anchors[0].1)?;
        let negative = self.embed(sentiment_anchors[1].1)?;
        let neutral = self.embed(sentiment_anchors[2].1)?;
        
        // Emotion anchors
        let emotion_anchors = Anchors::emotion_anchors();
        let joy = self.embed(emotion_anchors[0].1)?;
        let sadness = self.embed(emotion_anchors[1].1)?;
        let anger = self.embed(emotion_anchors[2].1)?;
        let fear = self.embed(emotion_anchors[3].1)?;
        let surprise = self.embed(emotion_anchors[4].1)?;
        let disgust = self.embed(emotion_anchors[5].1)?;
        
        // Theme anchors
        let theme_anchors = Anchors::default_theme_anchors();
        let mut theme_embeddings = Vec::new();
        
        for (name, text) in theme_anchors {
            let embedding = self.embed(text)?;
            theme_embeddings.push((name.to_string(), embedding));
        }
        
        println!("Anchor embeddings created successfully!");
        
        Ok(Anchors {
            positive,
            negative,
            neutral,
            joy,
            sadness,
            anger,
            fear,
            surprise,
            disgust,
            theme_embeddings,
        })
    }
}