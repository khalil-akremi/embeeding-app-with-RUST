use std::fs;
use eframe::egui;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Simple Embedding Demo",
        native_options,
        Box::new(|cc| Box::new(MyApp::new(cc))),
    )
}

struct MyApp {
    file_path: Option<String>,
    file_content: String,
    chunks: Vec<String>,
    embeddings: Vec<Vec<f32>>,  // We'll generate simple embeddings
    is_processing: bool,
    error_message: Option<String>,
    words_per_chunk: usize,
    embedding_dim: usize,
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
            embedding_dim: 16,
        }
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
    
    // SIMPLE embedding - NO Candle
    fn generate_simple_embeddings(&mut self, dimensions: usize) {
        self.embeddings.clear();
        self.is_processing = true;
        
        for (i, chunk) in self.chunks.iter().enumerate() {
            let mut embedding = vec![0.0; dimensions];
            
            // Fill based on deterministic char-derived values
            for (j, c) in chunk.chars().enumerate().take(dimensions) {
                embedding[j] = (c as u32 % 100) as f32 / 100.0;
            }
            
            // Add simple word hints
            if chunk.contains("the") { embedding[0] += 0.1; }
            if chunk.contains("and") { embedding[1] += 0.1; }
            if chunk.contains("for") { embedding[2] += 0.1; }
            
            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in embedding.iter_mut() { *v /= norm; }
            }
            
            self.embeddings.push(embedding);
            if i % 5 == 0 { println!("Processed {}/{} chunks", i + 1, self.chunks.len()); }
        }
        
        self.is_processing = false;
        if !self.embeddings.is_empty() {
            println!("Generated {} embeddings of dimension {}", self.embeddings.len(), dimensions);
        }
    }
    
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() { return 0.0; }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("📚 File Embedding Demo (no Candle)");
            ui.separator();
            
            // File selection
            if ui.button("📂 Select Text File").clicked() { self.pick_file(); }
            
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
                if ui.button("✂️ Create Chunks").clicked() { self.chunk_text(self.words_per_chunk); }
                
                // Embedding controls
                if !self.chunks.is_empty() {
                    ui.label(format!("✅ Created {} text chunks", self.chunks.len()));
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Embedding dimension:");
                        ui.add(egui::DragValue::new(&mut self.embedding_dim).speed(1).clamp_range(8..=256));
                    });
                    if ui.button("🧬 Generate Embeddings").clicked() {
                        self.generate_simple_embeddings(self.embedding_dim);
                    }
                }
                
                // Embeddings view
                if !self.embeddings.is_empty() {
                    ui.separator();
                    ui.heading("📊 Embeddings Generated");
                    ui.label(format!("Total: {}", self.embeddings.len()));
                    ui.label(format!("Dimension: {}", self.embeddings[0].len()));
                    
                    egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                        ui.collapsing("First embedding vector", |ui| {
                            ui.code(format!("{:.4?}", self.embeddings[0].iter().take(12).collect::<Vec<_>>()));
                            if self.embeddings[0].len() > 12 { ui.label("... (truncated)"); }
                        });
                    });
                    
                    if self.embeddings.len() > 1 {
                        let sim = self.cosine_similarity(&self.embeddings[0], &self.embeddings[1]);
                        ui.add(egui::ProgressBar::new(sim.max(0.0).min(1.0)).text(format!("{:.1}% similar", sim * 100.0)).desired_width(220.0));
                    }
                    
                    ui.separator();
                    if ui.button("🗑️ Clear All").clicked() { self.chunks.clear(); self.embeddings.clear(); }
                }
            } else {
                ui.separator();
                ui.label("ℹ️ Select a text file to begin");
            }
            
            if let Some(error) = &self.error_message {
                ui.separator();
                ui.colored_label(egui::Color32::RED, format!("❌ Error: {}", error));
            }
        });
    }
}