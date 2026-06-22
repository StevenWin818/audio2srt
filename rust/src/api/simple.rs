#[flutter_rust_bridge::frb(sync)] // 同步模式
pub fn greet(name: String) -> String {
    format!("Hello, {name}!")
}

#[flutter_rust_bridge::frb(init)]
pub fn init_app() {
    // 默认工具
    flutter_rust_bridge::setup_default_user_utils();
}
