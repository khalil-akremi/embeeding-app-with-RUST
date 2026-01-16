use std::fs;

// This is the main entry point of our app
fn main() -> eframe::Result<()> {
    // Native options: window settings
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 400.0]) // Larger window
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };
    
    // Run the app
    eframe::run_native(
        "File Embedding App",
        native_options,
        Box::new(|cc| {
            Box::new(MyApp::new(cc))
        }),
    )
}

// Our app state - this holds all the data our app needs
struct MyApp {
    file_path: Option<String>,      // Store the selected file path
    file_content: String,           // Store the file content
    error_message: Option<String>,  // Store errors to display
    is_loading: bool,               // Show loading state
}

impl MyApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            file_path: None,
            file_content: String::new(),
            error_message: None,
            is_loading: false,
        }
    }
    
    fn open_file_dialog(&mut self) {
        // Reset previous state
        self.error_message = None;
        self.is_loading = true;
    }
    
    // This will be called from the update method
    fn pick_and_load_file(&mut self) {
        // Use rfd::FileDialog to open a file picker
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Text files", &["txt", "md", "rs", "py", "js", "html", "css"])
            .add_filter("All files", &["*"])
            .pick_file() 
        {
            // Convert path to string
            let path_str = path.to_string_lossy().to_string();
            self.file_path = Some(path_str.clone());
            
            // Try to read the file
            match fs::read_to_string(&path) {
                Ok(content) => {
                    self.file_content = content;
                    self.error_message = None;
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to read file: {}", e));
                    self.file_content = String::new();
                }
            }
        } else {
            // User cancelled the dialog
            self.error_message = Some("No file selected".to_string());
        }
        
        self.is_loading = false;
    }
}

// Implement the eframe::App trait - this is REQUIRED
impl eframe::App for MyApp {
    // This function is called every frame (60 times per second)
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Show a loading overlay if needed
        if self.is_loading {
            egui::Area::new(egui::Id::new("loading"))
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.spinner();
                        ui.label("Selecting file...");
                    });
                });
        }
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("📁 File Embedding App");
            ui.separator();
            
            // File selection button
            if ui.button("📂 Select File").clicked() && !self.is_loading {
                self.open_file_dialog();
                // IMPORTANT: We call pick_and_load_file immediately
                // This will BLOCK the UI while file dialog is open
                self.pick_and_load_file();
            }
            
            // Show loading state
            if self.is_loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Loading file...");
                });
            }
            
            // Show selected file path
            if let Some(path) = &self.file_path {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("📄 Selected:");
                    ui.monospace(path.as_str());
                });
                
                // Show file info
                ui.horizontal(|ui| {
                    ui.label("📊 File info:");
                    ui.label(format!("Size: {} bytes", self.file_content.len()));
                    ui.label(format!("Lines: {}", self.file_content.lines().count()));
                });
                
                // Show file content in a scrollable area
                ui.separator();
                ui.label("📝 Content Preview:");
                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        // Show first 2000 characters or full content
                        let preview = if self.file_content.len() > 2000 {
                            format!("{}...\n\n[Truncated - showing first 2000 characters]", 
                                    &self.file_content[..2000])
                        } else {
                            self.file_content.clone()
                        };
                        
                        // Use monospace font for code/file content
                        ui.add(
                            egui::TextEdit::multiline(&mut preview.as_str())
                                .font(egui::TextStyle::Monospace)
                                .desired_rows(10)
                                .interactive(false) // Read-only
                        );
                    });
                
                // Embedding button (disabled for now)
                ui.separator();
                ui.add_enabled(false, egui::Button::new("🚀 Generate Embeddings"));
                
                // Clear button
                if ui.button("🗑️ Clear").clicked() {
                    self.file_path = None;
                    self.file_content = String::new();
                    self.error_message = None;
                }
            }
            
            // Show error messages if any
            if let Some(error) = &self.error_message {
                ui.separator();
                ui.colored_label(egui::Color32::RED, format!("❌ Error: {}", error));
                
                // Retry button on error
                if ui.button("🔄 Retry").clicked() {
                    self.error_message = None;
                }
            }
            
            // Show help text if no file selected
            if self.file_path.is_none() && !self.is_loading {
                ui.separator();
                ui.label("ℹ️  No file selected yet.");
                ui.label("Click 'Select File' to choose a text file.");
                ui.label("Supported: .txt, .md, .rs, .py, .js, .html, .css");
            }
        });
    }
}