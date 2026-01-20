use anyhow::Result;

#[derive(Debug, Clone)]
pub struct SentimentScore {
    pub positive: f32,
    pub negative: f32,
    pub neutral: f32,
}

impl SentimentScore {
    pub fn dominant(&self) -> &str {
        if self.positive > self.negative && self.positive > self.neutral {
            "Positive"
        } else if self.negative > self.positive && self.negative > self.neutral {
            "Negative"
        } else {
            "Neutral"
        }
    }
    
    pub fn confidence(&self) -> f32 {
        self.positive.max(self.negative).max(self.neutral)
    }
}

#[derive(Debug, Clone)]
pub struct EmotionScore {
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub surprise: f32,
    pub disgust: f32,
}

impl EmotionScore {
    pub fn top_emotions(&self, threshold: f32) -> Vec<(&str, f32)> {
        let mut emotions = vec![
            ("Joy", self.joy),
            ("Sadness", self.sadness),
            ("Anger", self.anger),
            ("Fear", self.fear),
            ("Surprise", self.surprise),
            ("Disgust", self.disgust),
        ];
        
        emotions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        emotions.into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect()
    }
    
    pub fn dominant(&self) -> (&str, f32) {
        let emotions = [
            ("Joy", self.joy),
            ("Sadness", self.sadness),
            ("Anger", self.anger),
            ("Fear", self.fear),
            ("Surprise", self.surprise),
            ("Disgust", self.disgust),
        ];
        
        emotions.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(name, score)| (*name, *score))
            .unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct ThemeScore {
    pub themes: Vec<(String, f32)>,
}

impl ThemeScore {
    pub fn top_themes(&self, threshold: f32) -> Vec<(&str, f32)> {
        self.themes.iter()
            .filter(|(_, score)| *score >= threshold)
            .map(|(name, score)| (name.as_str(), *score))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ChunkAnalysis {
    pub chunk_index: usize,
    pub sentiment: SentimentScore,
    pub emotion: EmotionScore,
    pub themes: ThemeScore,
}

pub struct Anchors {
    // Sentiment anchors
    pub positive: Vec<f32>,
    pub negative: Vec<f32>,
    pub neutral: Vec<f32>,
    
    // Emotion anchors
    pub joy: Vec<f32>,
    pub sadness: Vec<f32>,
    pub anger: Vec<f32>,
    pub fear: Vec<f32>,
    pub surprise: Vec<f32>,
    pub disgust: Vec<f32>,
    
    // Theme anchors (customizable)
    pub theme_embeddings: Vec<(String, Vec<f32>)>,
}

impl Anchors {
    pub fn sentiment_anchors() -> Vec<(&'static str, &'static str)> {
        vec![
            ("positive", "This is absolutely wonderful, excellent, amazing, fantastic, and great. I love it and feel very happy and delighted about this. Everything is perfect and outstanding."),
            ("negative", "This is terrible, awful, horrible, disgusting, and bad. I hate it and feel very upset and disappointed about this. Everything is wrong and unacceptable."),
            ("neutral", "This is okay, average, normal, standard, and moderate. It is neither good nor bad, just acceptable and adequate. Everything is fine and unremarkable."),
        ]
    }
    
    pub fn emotion_anchors() -> Vec<(&'static str, &'static str)> {
        vec![
            ("joy", "I am extremely happy, delighted, joyful, cheerful, and thrilled about this. I feel wonderful, excited, and full of happiness and pleasure."),
            ("sadness", "I feel very sad, disappointed, sorrowful, unhappy, and dejected about this. I am heartbroken, melancholic, and filled with grief and despair."),
            ("anger", "I am furious, angry, outraged, mad, and enraged about this. I feel irritated, frustrated, and filled with rage and hostility."),
            ("fear", "I am terrified, scared, afraid, anxious, and worried about this. I feel nervous, panicked, and filled with fear and dread."),
            ("surprise", "I am shocked, amazed, astonished, surprised, and stunned by this. I feel astounded, startled, and filled with wonder and disbelief."),
            ("disgust", "I find this repulsive, disgusting, revolting, nauseating, and gross. I feel sickened, appalled, and filled with revulsion and distaste."),
        ]
    }
    
    pub fn default_theme_anchors() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Technology", "Software development, programming, coding, algorithms, computers, digital technology, artificial intelligence, machine learning, data science, and technical systems."),
            ("Business", "Revenue, profits, strategy, market growth, sales, business operations, finance, investments, corporate management, and commercial activities."),
            ("Health", "Wellness, medicine, fitness, healthcare, medical treatment, mental health, exercise, nutrition, physical wellbeing, and healthy lifestyle."),
            ("Education", "Learning, teaching, schools, universities, students, academic studies, knowledge, training, educational programs, and intellectual development."),
            ("Science", "Research, experiments, scientific discovery, physics, chemistry, biology, scientific method, data analysis, laboratory work, and empirical studies."),
            ("Entertainment", "Movies, music, games, television, sports, recreation, leisure activities, fun, enjoyment, and entertainment media."),
        ]
    }
}

pub struct SemanticAnalyzer;

impl SemanticAnalyzer {
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
    
    pub fn analyze_sentiment(embedding: &[f32], anchors: &Anchors) -> SentimentScore {
        let positive = Self::cosine_similarity(embedding, &anchors.positive);
        let negative = Self::cosine_similarity(embedding, &anchors.negative);
        let neutral = Self::cosine_similarity(embedding, &anchors.neutral);
        
        // Normalize scores to sum to 1.0
        let total = positive + negative + neutral;
        
        SentimentScore {
            positive: if total > 0.0 { positive / total } else { 0.33 },
            negative: if total > 0.0 { negative / total } else { 0.33 },
            neutral: if total > 0.0 { neutral / total } else { 0.34 },
        }
    }
    
    pub fn analyze_emotion(embedding: &[f32], anchors: &Anchors) -> EmotionScore {
        let joy = Self::cosine_similarity(embedding, &anchors.joy);
        let sadness = Self::cosine_similarity(embedding, &anchors.sadness);
        let anger = Self::cosine_similarity(embedding, &anchors.anger);
        let fear = Self::cosine_similarity(embedding, &anchors.fear);
        let surprise = Self::cosine_similarity(embedding, &anchors.surprise);
        let disgust = Self::cosine_similarity(embedding, &anchors.disgust);
        
        // Normalize
        let total = joy + sadness + anger + fear + surprise + disgust;
        
        EmotionScore {
            joy: if total > 0.0 { joy / total } else { 0.16 },
            sadness: if total > 0.0 { sadness / total } else { 0.16 },
            anger: if total > 0.0 { anger / total } else { 0.17 },
            fear: if total > 0.0 { fear / total } else { 0.17 },
            surprise: if total > 0.0 { surprise / total } else { 0.17 },
            disgust: if total > 0.0 { disgust / total } else { 0.17 },
        }
    }
    
    pub fn analyze_themes(embedding: &[f32], anchors: &Anchors) -> ThemeScore {
        let mut themes = Vec::new();
        
        for (theme_name, theme_embedding) in &anchors.theme_embeddings {
            let similarity = Self::cosine_similarity(embedding, theme_embedding);
            themes.push((theme_name.clone(), similarity));
        }
        
        // Sort by score descending
        themes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        ThemeScore { themes }
    }
    
    pub fn analyze_chunk(
        chunk_index: usize,
        embedding: &[f32],
        anchors: &Anchors,
    ) -> ChunkAnalysis {
        ChunkAnalysis {
            chunk_index,
            sentiment: Self::analyze_sentiment(embedding, anchors),
            emotion: Self::analyze_emotion(embedding, anchors),
            themes: Self::analyze_themes(embedding, anchors),
        }
    }
    
    pub fn analyze_all_chunks(
        embeddings: &[Vec<f32>],
        anchors: &Anchors,
    ) -> Vec<ChunkAnalysis> {
        embeddings.iter()
            .enumerate()
            .map(|(i, emb)| Self::analyze_chunk(i, emb, anchors))
            .collect()
    }
}

// Document-level aggregation
pub struct DocumentAnalysis {
    pub overall_sentiment: SentimentScore,
    pub overall_emotion: EmotionScore,
    pub theme_distribution: Vec<(String, f32)>,
    pub chunk_count: usize,
}

impl DocumentAnalysis {
    pub fn from_chunks(analyses: &[ChunkAnalysis]) -> Self {
        if analyses.is_empty() {
            return Self {
                overall_sentiment: SentimentScore { positive: 0.33, negative: 0.33, neutral: 0.34 },
                overall_emotion: EmotionScore { joy: 0.16, sadness: 0.16, anger: 0.17, fear: 0.17, surprise: 0.17, disgust: 0.17 },
                theme_distribution: Vec::new(),
                chunk_count: 0,
            };
        }
        
        let count = analyses.len() as f32;
        
        // Average sentiment
        let overall_sentiment = SentimentScore {
            positive: analyses.iter().map(|a| a.sentiment.positive).sum::<f32>() / count,
            negative: analyses.iter().map(|a| a.sentiment.negative).sum::<f32>() / count,
            neutral: analyses.iter().map(|a| a.sentiment.neutral).sum::<f32>() / count,
        };
        
        // Average emotion
        let overall_emotion = EmotionScore {
            joy: analyses.iter().map(|a| a.emotion.joy).sum::<f32>() / count,
            sadness: analyses.iter().map(|a| a.emotion.sadness).sum::<f32>() / count,
            anger: analyses.iter().map(|a| a.emotion.anger).sum::<f32>() / count,
            fear: analyses.iter().map(|a| a.emotion.fear).sum::<f32>() / count,
            surprise: analyses.iter().map(|a| a.emotion.surprise).sum::<f32>() / count,
            disgust: analyses.iter().map(|a| a.emotion.disgust).sum::<f32>() / count,
        };
        
        // Theme distribution (average scores per theme)
        let mut theme_map: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        
        for analysis in analyses {
            for (theme, score) in &analysis.themes.themes {
                *theme_map.entry(theme.clone()).or_insert(0.0) += score;
            }
        }
        
        let mut theme_distribution: Vec<(String, f32)> = theme_map
            .into_iter()
            .map(|(theme, total)| (theme, total / count))
            .collect();
        
        theme_distribution.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Self {
            overall_sentiment,
            overall_emotion,
            theme_distribution,
            chunk_count: analyses.len(),
        }
    }
}