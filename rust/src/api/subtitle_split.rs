//! 字幕分条: 标点 + 停顿生成候选边界, 动态规划生成 2~5 秒字幕。
//!
//! 方案 (产品约定):
//!   1. Silero VAD 切成 20~45 秒语音块
//!   2. Qwen3-ASR 对每块输出完整带标点文本
//!   3. ForcedAligner 获取词级/字级时间戳 (当前为线性平分, 接口不变)
//!   4. 标点 + 停顿生成候选边界
//!   5. 动态规划生成 2~5 秒字幕
//!   6. 字幕最多两行 (偏好一行)
//!   7. 开始/结束时间轻微视觉平滑
//!   8. 输出 SRT/VTT (调用方)

use super::silero_vad::{TranscriptionSegment, WordItem};

/// 字幕最短时长 (ms)
const SUB_MIN_MS: i64 = 2000;
/// 字幕最长时长 (ms)
const SUB_MAX_MS: i64 = 5000;
/// 目标时长 (ms), 靠近该值代价最小
const SUB_TARGET_MS: i64 = 3500;
/// 两行排版上限 (字符数, 2 × 30)
const MAX_CHARS_2LINES: usize = 60;
/// 单行排版偏好 (字符数)
const PREFER_CHARS_1LINE: usize = 30;
/// 开始时间轻微视觉平滑 (ms)
const START_PAD_MS: i64 = 150;
/// 结束时间轻微视觉平滑 (ms)
const END_PAD_MS: i64 = 200;
/// 词间停顿超过该值视为强边界 (ms)
const PAUSE_STRONG_MS: i64 = 250;

fn is_strong_boundary(ch: char) -> bool {
    matches!(ch, '。' | '！' | '？' | '!' | '?' | '…' | '…' | '.')
}

fn is_weak_boundary(ch: char) -> bool {
    matches!(
        ch,
        '，' | '；' | '、' | ',' | ';' | '：' | ':' | ')' | '）' | '」' | '』' | '"' | '"' | '"'
    )
}

/// 将一段带标点文本及其词/字级时间戳拆成 2~5 秒的字幕 (动态规划)。
///
/// - `units` 为空时返回空 Vec, 调用方回退为整段一条字幕;
/// - 拆分点优先落在句末标点/长停顿, 其次句中逗号, 无标点时按时长硬切;
/// - 输出字幕的时间已做轻微视觉平滑且保证相邻不重叠。
pub fn split_subtitles(units: &[WordItem]) -> Vec<TranscriptionSegment> {
    let n = units.len();
    if n == 0 {
        return Vec::new();
    }

    // dp[i] = 把 units[0..i) 拆成若干字幕的最小代价; prev[i] = 最后一段的起点
    let mut dp = vec![f64::INFINITY; n + 1];
    let mut prev = vec![0usize; n + 1];
    dp[0] = 0.0;

    for i in 1..=n {
        for j in 0..i {
            let total = dp[j] + segment_cost(units, j, i);
            if total < dp[i] {
                dp[i] = total;
                prev[i] = j;
            }
        }
    }

    // 回溯切分点
    let mut bounds = vec![n];
    let mut cur = n;
    while cur > 0 {
        cur = prev[cur];
        bounds.push(cur);
    }
    bounds.reverse();

    // 组装字幕 + 轻微视觉平滑 (前垫/后延, 不重叠, 不越出块边界)
    let block_start = units[0].start_ms;
    let block_end = units[n - 1].end_ms;
    let mut out = Vec::with_capacity(bounds.len() - 1);
    let mut prev_end = block_start;
    for w in bounds.windows(2) {
        let (j, i) = (w[0], w[1]);
        if j >= i {
            continue;
        }
        let text: String = units[j..i].iter().map(|u| u.text.as_str()).collect();
        let mut start = units[j].start_ms - START_PAD_MS;
        let mut end = units[i - 1].end_ms + END_PAD_MS;
        if start < block_start {
            start = block_start;
        }
        if end > block_end {
            end = block_end;
        }
        if start < prev_end {
            start = prev_end;
        }
        if end <= start {
            end = start + 1;
        }
        prev_end = end;
        out.push(TranscriptionSegment {
            start_ms: start,
            end_ms: end,
            text,
            words: units[j..i].to_vec(),
            timestamp_quality: "Qwen3-DP-Split".to_string(),
        });
    }
    out
}

/// 一段字幕 (units[j..i)) 的拆分代价: 时长越贴近目标越好, 边界越接近
/// 句末标点/停顿越好, 字数越少越好 (最多两行, 偏好一行)。
fn segment_cost(units: &[WordItem], j: usize, i: usize) -> f64 {
    let dur = units[i - 1].end_ms - units[j].start_ms;
    let mut cost = 0.0;

    // 时长约束: 2~5 秒, 超限重罚, 目标 3.5s 附近轻偏好
    if dur < SUB_MIN_MS {
        cost += (SUB_MIN_MS - dur) as f64 * 0.02;
    }
    if dur > SUB_MAX_MS {
        cost += (dur - SUB_MAX_MS) as f64 * 0.15;
    }
    cost += (dur - SUB_TARGET_MS).abs() as f64 * 0.004;

    // 字数约束: 两行上限硬罚, 单行偏好轻罚
    let chars: usize = units[j..i].iter().map(|u| u.text.chars().count()).sum();
    if chars > MAX_CHARS_2LINES {
        cost += (chars - MAX_CHARS_2LINES) as f64 * 2.0;
    } else if chars > PREFER_CHARS_1LINE {
        cost += (chars - PREFER_CHARS_1LINE) as f64 * 0.3;
    }

    // 边界质量: 句末标点/长停顿 > 句中逗号 > 无标点
    if j > 0 {
        let prev_ch = units[j - 1].text.chars().last().unwrap_or(' ');
        let pause = units[j].start_ms - units[j - 1].end_ms;
        cost += if is_strong_boundary(prev_ch) || pause >= PAUSE_STRONG_MS {
            0.0
        } else if is_weak_boundary(prev_ch) {
            2.0
        } else {
            8.0
        };
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units_from_text(text: &str, start_ms: i64, dur_per_char_ms: i64) -> Vec<WordItem> {
        text.chars()
            .enumerate()
            .map(|(i, c)| WordItem {
                text: c.to_string(),
                start_ms: start_ms + i as i64 * dur_per_char_ms,
                end_ms: start_ms + (i as i64 + 1) * dur_per_char_ms,
                confidence: 1.0,
            })
            .collect()
    }

    #[test]
    fn empty_units_returns_empty() {
        assert!(split_subtitles(&[]).is_empty());
    }

    #[test]
    fn short_block_stays_one_subtitle() {
        let units = units_from_text("你好世界。", 0, 400);
        let segs = split_subtitles(&units);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "你好世界。");
    }

    #[test]
    fn long_block_splits_at_punctuation() {
        // 72 字 × 100ms = 7.2s > 5s 上限, 应拆成多段 (边界优先落在句号后)
        let text = "今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。";
        let units = units_from_text(text, 0, 100);
        let segs = split_subtitles(&units);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        assert!(segs.iter().all(|s| s.end_ms - s.start_ms <= 5200));
        assert!(segs.iter().all(|s| s.end_ms - s.start_ms >= 1500));
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn times_are_monotonic_and_smoothed() {
        let text = "第一句。第二句。第三句。第四句。第五句。第六句。第七句。第八句。第九句。第十句。";
        let units = units_from_text(text, 1000, 150);
        let segs = split_subtitles(&units);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        for w in segs.windows(2) {
            assert!(w[1].start_ms >= w[0].end_ms, "overlap detected");
        }
        assert!(segs[0].start_ms >= 1000);
        assert!(segs.last().unwrap().end_ms <= units.last().unwrap().end_ms);
    }
}
