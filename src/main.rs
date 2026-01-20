use std::fs;
use eframe::egui;

mod embedder;
mod analyzers;

use embedder::MiniLMEmbedder;
use analyzers::{Anchors, SemanticAnalyzer, ChunkAnalysis, DocumentAnalysis};

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "MiniLM Embedding Demo",
        native_options,
        Box::new(|cc| Box::new(MyApp::new(cc))),
    )
}

struct MyApp {
    file_path: Option<String>,
    file_content: String,
    chunks: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    is_processing: bool,
    error_message: Option<String>,
    words_per_chunk: usize,
    embedder: Option<MiniLMEmbedder>,
    model_loading: bool,
    model_loaded: bool,
    anchors: Option<Anchors>,
    chunk_analyses: Vec<ChunkAnalysis>,
    doc_analysis: Option<DocumentAnalysis>,
    emotion_threshold: f32,
    theme_threshold: f32,
}

impl MyApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            file_path: None,
            file_content: String::new(),
            chunks: Vec::new(),
            embeddings: Vec::new(),
            is_processing: false,
            error_message: None,
            words_per_chunk: 50,
            embedder: None,
            model_loading: false,
            model_loaded: false,
            anchors: None,
            chunk_analyses: Vec::new(),
            doc_analysis: None,
            emotion_threshold: 0.20,
            theme_threshold: 0.30,
        }
    }
    
    fn load_model(&mut self) {
        self.model_loading = true;
        self.error_message = None;
        
        match MiniLMEmbedder::new() {
            Ok(embedder) => {
                // Create anchors
                match embedder.create_anchors() {
                    Ok(anchors) => {
                        self.anchors = Some(anchors);
                        self.embedder = Some(embedder);
                        self.model_loaded = true;
                        self.error_message = Some("✅ Model and anchors loaded successfully!".to_string());
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Failed to create anchors: {}", e));
                    }
                }
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load model: {}", e));
            }
        }
        
        self.model_loading = false;
    }
    
    fn pick_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Text files", &["txt", "md", "rs", "py", "js"])
            .pick_file()
        {
            self.file_path = Some(path.to_string_lossy().to_string());
            
            match fs::read_to_string(&path) {
                Ok(content) => {
                    self.file_content = content;
                    self.error_message = None;
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to read file: {}", e));
                }
            }
        }
    }
    
    fn chunk_text(&mut self, words_per_chunk: usize) {
        self.chunks.clear();
        
        let words: Vec<&str> = self.file_content.split_whitespace().collect();
        let mut start = 0;
        
        while start < words.len() {
            let end = (start + words_per_chunk).min(words.len());
            let chunk = words[start..end].join(" ");
            self.chunks.push(chunk);
            start = end;
        }
        
        println!("Created {} chunks", self.chunks.len());
    }
    
    fn generate_embeddings(&mut self) {
        if let Some(embedder) = &self.embedder {
            self.is_processing = true;
            self.embeddings.clear();
            
            match embedder.embed_batch(&self.chunks) {
                Ok(embeddings) => {
                    self.embeddings = embeddings;
                    self.error_message = Some(format!("✅ Generated {} embeddings", self.embeddings.len()));
                    println!("Successfully generated {} embeddings", self.embeddings.len());
                    
                    // Automatically run analysis after embedding
                    self.analyze_chunks();
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to generate embeddings: {}", e));
                }
            }
            
            self.is_processing = false;
        } else {
            self.error_message = Some("Please load the model first!".to_string());
        }
    }
    
    fn analyze_chunks(&mut self) {
        if let Some(anchors) = &self.anchors {
            println!("Analyzing chunks...");
            self.chunk_analyses = SemanticAnalyzer::analyze_all_chunks(&self.embeddings, anchors);
            self.doc_analysis = Some(DocumentAnalysis::from_chunks(&self.chunk_analyses));
            println!("Analysis complete!");
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("📚 MiniLM-L6-v2 Embedding Demo");
            ui.separator();
            
            // Model loading section
            ui.horizontal(|ui| {
                if !self.model_loaded {
                    if ui.button("🔧 Load MiniLM Model").clicked() && !self.model_loading {
                        self.load_model();
                    }
                    if self.model_loading {
                        ui.spinner();
                        ui.label("Loading model...");
                    }
                } else {
                    ui.colored_label(egui::Color32::GREEN, "✅ Model Ready");
                }
            });
            
            ui.separator();
            
            // File selection (only enabled if model is loaded)
            ui.add_enabled_ui(self.model_loaded, |ui| {
                if ui.button("📂 Select Text File").clicked() {
                    self.pick_file();
                }
                
                if let Some(path) = &self.file_path {
                    ui.label(format!("📄 Selected: {}", path));
                    ui.horizontal(|ui| {
                        ui.label(format!("Size: {} chars", self.file_content.len()));
                        ui.label(format!("Words: {}", self.file_content.split_whitespace().count()));
                    });
                    
                    // Chunking controls
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Split into chunks of");
                        ui.add(egui::DragValue::new(&mut self.words_per_chunk).speed(1).clamp_range(10..=200));
                        ui.label("words");
                    });
                    
                    if ui.button("✂️ Create Chunks").clicked() {
                        self.chunk_text(self.words_per_chunk);
                    }
                    
                    // Embedding controls
                    if !self.chunks.is_empty() {
                        ui.label(format!("✅ Created {} text chunks", self.chunks.len()));
                        ui.separator();
                        
                        ui.horizontal(|ui| {
                            if ui.button("🧬 Generate Embeddings").clicked() && !self.is_processing {
                                self.generate_embeddings();
                            }
                            if self.is_processing {
                                ui.spinner();
                                ui.label("Processing...");
                            }
                        });
                    }
                    
                    // Embeddings view
                    if !self.embeddings.is_empty() {
                        ui.separator();
                        ui.heading("📊 Embeddings Generated");
                        ui.label(format!("Total: {}", self.embeddings.len()));
                        ui.label(format!("Dimension: {} (all-MiniLM-L6-v2 standard)", self.embeddings[0].len()));
                        
                        egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                            ui.collapsing("First embedding vector", |ui| {
                                ui.code(format!("{:.4?}", self.embeddings[0].iter().take(12).collect::<Vec<_>>()));
                                if self.embeddings[0].len() > 12 {
                                    ui.label("... (truncated)");
                                }
                            });
                            
                            ui.collapsing("First chunk text", |ui| {
                                let preview = if self.chunks[0].len() > 200 {
                                    format!("{}...", &self.chunks[0][..200])
                                } else {
                                    self.chunks[0].clone()
                                };
                                ui.label(preview);
                            });
                        });
                        
                        // Similarity comparison
                        if self.embeddings.len() > 1 {
                            ui.separator();
                            ui.label("Similarity between first two chunks:");
                            if let Some(embedder) = &self.embedder {
                                let sim = embedder.cosine_similarity(&self.embeddings[0], &self.embeddings[1]);
                                ui.add(
                                    egui::ProgressBar::new(sim.max(0.0).min(1.0))
                                        .text(format!("Cosine similarity: {:.1}%", sim * 100.0))
                                        .desired_width(300.0)
                                );
                            }
                        }
                        
                        ui.separator();
                        if ui.button("🗑️ Clear All").clicked() {
                            self.chunks.clear();
                            self.embeddings.clear();
                            self.chunk_analyses.clear();
                            self.doc_analysis = None;
                        }
                    }
                    
                    // Analysis Results
                    if !self.chunk_analyses.is_empty() {
                        ui.separator();
                        ui.heading("🔍 Semantic Analysis Results");
                        
                        // Document-level summary
                        if let Some(doc_analysis) = &self.doc_analysis {
                            ui.separator();
                            ui.label("📊 Document Overview:");
                            
                            egui::Grid::new("doc_stats")
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label("Chunks analyzed:");
                                    ui.label(format!("{}", doc_analysis.chunk_count));
                                    ui.end_row();
                                    
                                    ui.label("Overall sentiment:");
                                    let sent = &doc_analysis.overall_sentiment;
                                    ui.label(format!("{} ({:.1}%)", 
                                        sent.dominant(), 
                                        sent.confidence() * 100.0
                                    ));
                                    ui.end_row();
                                    
                                    ui.label("Sentiment breakdown:");
                                    ui.horizontal(|ui| {
                                        ui.colored_label(egui::Color32::GREEN, 
                                            format!("Pos: {:.1}%", sent.positive * 100.0));
                                        ui.colored_label(egui::Color32::RED, 
                                            format!("Neg: {:.1}%", sent.negative * 100.0));
                                        ui.colored_label(egui::Color32::GRAY, 
                                            format!("Neu: {:.1}%", sent.neutral * 100.0));
                                    });
                                    ui.end_row();
                                    
                                    let (emotion, score) = doc_analysis.overall_emotion.dominant();
                                    ui.label("Dominant emotion:");
                                    ui.label(format!("{} ({:.1}%)", emotion, score * 100.0));
                                    ui.end_row();
                                    
                                    ui.label("Top themes:");
                                    ui.vertical(|ui| {
                                        for (theme, score) in doc_analysis.theme_distribution.iter().take(3) {
                                            ui.label(format!("• {} ({:.1}%)", theme, score * 100.0));
                                        }
                                    });
                                    ui.end_row();
                                });
                        }
                        
                        ui.separator();
                        
                        // Threshold controls
                        ui.horizontal(|ui| {
                            ui.label("Emotion threshold:");
                            ui.add(egui::Slider::new(&mut self.emotion_threshold, 0.0..=0.5)
                                .text("%")
                                .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)));
                            
                            ui.separator();
                            
                            ui.label("Theme threshold:");
                            ui.add(egui::Slider::new(&mut self.theme_threshold, 0.0..=0.6)
                                .text("%")
                                .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)));
                        });
                        
                        ui.separator();
                        
                        // Per-chunk analysis
                        egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                            for analysis in &self.chunk_analyses {
                                ui.collapsing(format!("Chunk {} Analysis", analysis.chunk_index + 1), |ui| {
                                    // Show chunk text preview
                                    if let Some(chunk_text) = self.chunks.get(analysis.chunk_index) {
                                        let preview = if chunk_text.len() > 150 {
                                            format!("{}...", &chunk_text[..150])
                                        } else {
                                            chunk_text.clone()
                                        };
                                        ui.label(format!("📝 Text: {}", preview));
                                        ui.separator();
                                    }
                                    
                                    // Sentiment
                                    let sent = &analysis.sentiment;
                                    ui.label(format!("😊 Sentiment: {} ({:.1}%)", 
                                        sent.dominant(), 
                                        sent.confidence() * 100.0
                                    ));
                                    ui.horizontal(|ui| {
                                        ui.add(egui::ProgressBar::new(sent.positive)
                                            .text(format!("Pos: {:.1}%", sent.positive * 100.0))
                                            .fill(egui::Color32::GREEN));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.add(egui::ProgressBar::new(sent.negative)
                                            .text(format!("Neg: {:.1}%", sent.negative * 100.0))
                                            .fill(egui::Color32::RED));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.add(egui::ProgressBar::new(sent.neutral)
                                            .text(format!("Neu: {:.1}%", sent.neutral * 100.0))
                                            .fill(egui::Color32::GRAY));
                                    });
                                    
                                    ui.separator();
                                    
                                    // Emotions
                                    let emotions = analysis.emotion.top_emotions(self.emotion_threshold);
                                    if !emotions.is_empty() {
                                        ui.label(format!("🎭 Emotions (>{:.0}% threshold):", self.emotion_threshold * 100.0));
                                        for (emotion, score) in emotions {
                                            ui.label(format!("  • {}: {:.1}%", emotion, score * 100.0));
                                        }
                                    } else {
                                        ui.label("🎭 Emotions: None above threshold");
                                    }
                                    
                                    ui.separator();
                                    
                                    // Themes
                                    let themes = analysis.themes.top_themes(self.theme_threshold);
                                    if !themes.is_empty() {
                                        ui.label(format!("🏷️ Themes (>{:.0}% threshold):", self.theme_threshold * 100.0));
                                        for (theme, score) in themes {
                                            ui.label(format!("  • {}: {:.1}%", theme, score * 100.0));
                                        }
                                    } else {
                                        ui.label("🏷️ Themes: None above threshold");
                                    }
                                });
                            }
                        });
                        
                        ui.separator();
                        if ui.button("🗑️ Clear Analysis").clicked() {
                            self.chunk_analyses.clear();
                            self.doc_analysis = None;
                        }
                    }
                } else if self.model_loaded {
                    ui.separator();
                    ui.label("ℹ️ Select a text file to begin");
                }
            });
            
            if let Some(msg) = &self.error_message {
                ui.separator();
                let color = if msg.starts_with("✅") {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::RED
                };
                ui.colored_label(color, msg);
            }
        });
    }
}