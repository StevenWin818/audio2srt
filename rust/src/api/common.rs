use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WordItem {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub words: Vec<WordItem>,
    pub timestamp_quality: String,
}

#[derive(Clone, Debug)]
pub enum TranscriptionEvent {
    Progress(i32),
    ProgressDetail { processed_ms: i64, total_ms: i64 },
    Success(Vec<TranscriptionSegment>),
    Failure(String),
    Segment(TranscriptionSegment),
}

pub fn convert_chinese(text: String, to_simplified: bool) -> String {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    zhconv::zhconv(&text, target)
}

pub fn convert_chinese_list(texts: Vec<String>, to_simplified: bool) -> Vec<String> {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    texts.into_iter().map(|text| zhconv::zhconv(&text, target)).collect()
}

#[cfg(target_os = "windows")]
pub fn register_thread_as_pro_audio() {
    use windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsA;

    thread_local! {
        static MMCSS_HANDLE: std::cell::RefCell<Option<usize>> = const { std::cell::RefCell::new(None) };
    }

    MMCSS_HANDLE.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            let task_name = std::ffi::CString::new("Pro Audio").unwrap();
            let mut task_index: u32 = 0;
            let handle = unsafe {
                AvSetMmThreadCharacteristicsA(task_name.as_ptr() as *const u8, &mut task_index)
            };
            if handle != std::ptr::null_mut() {
                *guard = Some(handle as usize);
            }
        }
    });
}

#[cfg(not(target_os = "windows"))]
pub fn register_thread_as_pro_audio() {}
