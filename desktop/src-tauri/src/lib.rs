// Hands — the desktop shell.
//
// The window is a skin: every value the UI needs comes from the Persona backend
// over plain HTTP/WebSocket, so this process owns no business logic and holds no
// credentials. The persona key stays inside the webview's own storage.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// Reports the shell build, so the UI can prove it is running inside the
/// native exe rather than a browser tab.
#[tauri::command]
fn shell_info(app: tauri::AppHandle) -> serde_json::Value {
    let version = app.package_info().version.to_string();
    serde_json::json!({
        "name": "hands-desktop",
        "version": version,
        "platform": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![shell_info])
        .run(tauri::generate_context!())
        .expect("Hands failed to start");
}