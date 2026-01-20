use std::fs;
use eframe::egui;
mod simple_model;
use simple_model::MiniLMEmbedder;



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
        }
    }
    
    fn load_model(&mut self) {
        self.model_loading = true;
        self.error_message = None;
        
        match MiniLMEmbedder::new() {
            Ok(embedder) => {
                self.embedder = Some(embedder);
                self.model_loaded = true;
                self.error_message = Some("✅ Model loaded successfully!".to_string());
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