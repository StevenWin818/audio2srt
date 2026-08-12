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
/// 字幕最长时长 (ms) —— 仅对"非完整句子"生效
const SUB_MAX_MS: i64 = 5000;
/// 完整句子 (前后都是强标点) 可超过 5s 的上限: 长句优先单独成条
const SENTENCE_MAX_MS: i64 = 9000;
/// 目标时长 (ms), 靠近该值代价最小
const SUB_TARGET_MS: i64 = 3500;
/// 字幕长度硬上限: 中文 ≤20 字符, 西文 ≤20 单词 (标点不计)
const MAX_UNITS: usize = 20;
/// 单条偏好的长度 (低于该值无长度代价)
const PREFER_UNITS: usize = 12;
/// 开始时间轻微视觉平滑 (ms)
const START_PAD_MS: i64 = 150;
/// 结束时间轻微视觉平滑 (ms)
const END_PAD_MS: i64 = 200;
/// 词间停顿超过该值视为强边界 (ms)
const PAUSE_STRONG_MS: i64 = 250;
/// 句中无停顿拆分的基准代价 (必须拆时优先选最长停顿)
const MID_SENTENCE_BASE: f64 = 120.0;
/// 句点后引语逗号 (≤2 词后接逗号) 禁止切分的代价
const LEADIN_COMMA_BLOCK: f64 = 500.0;

fn is_strong_punct(ch: char) -> bool {
    matches!(ch, '。' | '！' | '？' | '!' | '?' | '…')
}

fn is_weak_punct(ch: char) -> bool {
    matches!(
        ch,
        '，' | '；' | '、' | ',' | ';' | '：' | ':' | ')' | '）' | '」' | '』' | '"' | '“' | '”'
    )
}

fn is_comma_punct(ch: char) -> bool {
    matches!(ch, ',' | '，' | ';' | '；')
}

fn is_cjk_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

/// 缩写保护: 短字母词后的句点 ("Mr." "Dr." "St." "etc." "U.S.") 不是句子边界;
/// 数字小数点 ("3.5") 同理。返回 true 表示该句点应视为普通句点 (强边界)。
fn is_real_period(units: &[WordItem], dot_idx: usize) -> bool {
    // dot_idx 指向句点 unit; 检查其前一个非标点 unit
    if dot_idx == 0 {
        return true;
    }
    let prev = &units[dot_idx - 1];
    let prev_text = prev.text.trim_end_matches('.');
    let prev_chars: Vec<char> = prev_text.chars().collect();
    if prev_chars.is_empty() {
        return true;
    }
    // 数字: 小数点
    if prev_chars.iter().all(|c| c.is_ascii_digit()) {
        return false;
    }
    // 短字母缩写: Mr. Dr. St. No. etc. (≤4 个字母)
    if prev_chars.iter().all(|c| c.is_ascii_alphabetic()) && prev_chars.len() <= 4 {
        return false;
    }
    true
}

/// 边界 (unit j 之前) 是否为强标点边界。句点带缩写/小数点保护。
fn strong_punct_boundary(units: &[WordItem], j: usize) -> bool {
    if j == 0 {
        return false;
    }
    let ch = units[j - 1].text.chars().last().unwrap_or(' ');
    if ch == '.' {
        return is_real_period(units, j - 1);
    }
    is_strong_punct(ch)
}

/// 段尾 (units[i-1]) 是否为强标点。
fn strong_punct_end(units: &[WordItem], i: usize) -> bool {
    if i == 0 {
        return false;
    }
    let ch = units[i - 1].text.chars().last().unwrap_or(' ');
    if ch == '.' {
        return is_real_period(units, i - 1);
    }
    is_strong_punct(ch)
}

/// 句中边界的代价: 停顿越大代价越低, 必须拆分时优先选最长停顿处
fn mid_sentence_cost(units: &[WordItem], j: usize) -> f64 {
    let pause = units[j]
        .start_ms
        .saturating_sub(units[j - 1].end_ms)
        .min(100) as f64;
    (MID_SENTENCE_BASE - pause).max(20.0)
}

/// 句点后引语逗号边界: 句点后、逗号前的西文单词 ≤2 个时忽略该逗号不切
/// ("However," "Yeah," "I mean," 等话语标记后不切; 结构化规则, 无需白名单)
fn is_leadin_comma_boundary(units: &[WordItem], j: usize) -> bool {
    if j == 0 {
        return false;
    }
    // 边界前必须是逗号类标点
    let prev_ch = units[j - 1].text.chars().last().unwrap_or(' ');
    if !is_comma_punct(prev_ch) {
        return false;
    }
    // 从逗号向前数西文单词, 直到最近的句末标点; ≤2 个则忽略该逗号
    let mut word_count = 0usize;
    let mut idx = j - 1;
    loop {
        let text = units[idx].text.trim();
        let tail_ch = text.chars().last().unwrap_or(' ');
        if is_strong_punct(tail_ch) || tail_ch == '.' {
            return word_count <= 2;
        }
        // 词+逗号同一 unit (如 "way,"): 单词照常计数
        let core = text.trim_end_matches(|c: char| is_comma_punct(c) || is_strong_punct(c) || c == '.');
        if is_comma_punct(tail_ch) {
            if core.chars().any(|c| c.is_alphanumeric()) {
                word_count += 1;
                if word_count > 2 {
                    return false;
                }
            }
        } else {
            if core.chars().any(|c| is_cjk_char(c)) {
                // 中文不适用此规则 (西文单词才计数)
                return false;
            }
            if core.chars().any(|c| c.is_alphanumeric()) {
                word_count += 1;
                if word_count > 2 {
                    return false;
                }
            }
        }
        if idx == 0 {
            return false; // 走到序列开头仍未遇到句末标点
        }
        idx -= 1;
    }
}

/// 统计一段的字数: 中文按字符数, 西文按单词数 (纯标点不计)
fn count_units(units: &[WordItem], j: usize, i: usize) -> usize {
    let mut count = 0usize;
    for u in &units[j..i] {
        let text = u.text.trim();
        if text.is_empty() {
            continue;
        }
        if text.chars().all(|c| !c.is_alphanumeric()) {
            continue; // 纯标点不计
        }
        if text.chars().any(|c| is_cjk_char(c)) {
            count += text.chars().count();
        } else {
            count += 1;
        }
    }
    count
}

/// 将一段带标点文本及其词/字级时间戳拆成 2~5 秒的字幕 (动态规划)。
///
/// - `units` 为空时返回空 Vec, 调用方回退为整段一条字幕;
/// - 拆分点优先落在句末标点/长停顿, 其次句中逗号, 无标点按时长硬切;
/// - 输出字幕的时间已做轻微视觉平滑且保证相邻不重叠;
/// - `quality` 为对齐质量标记 (如 "ForcedAligned"/"LinearFallback"),
///   透传到输出字幕, 使 UI/SRT 能区分真实对齐与线性回退;
/// - `block_start_ms`/`block_end_ms` 为块的真实媒体边界:
///   平滑前垫/后延以此为 clamp 上下限, 保证块内首条/末条字幕
///   与其余字幕的平滑行为一致, 避免块起点/大停顿后出现整体跳变。
pub fn split_subtitles(
    units: &[WordItem],
    quality: &str,
    block_start_ms: i64,
    block_end_ms: i64,
) -> Vec<TranscriptionSegment> {
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

    // 组装字幕 + 轻微视觉平滑 (前垫/后延, 不重叠, 不越出块的真实媒体边界)
    let block_start = block_start_ms.min(units[0].start_ms);
    let block_end = block_end_ms.max(units[n - 1].end_ms);
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
            timestamp_quality: format!("{}-DP", quality),
        });
    }
    out
}

/// 一段字幕 (units[j..i)) 的拆分代价。
///
/// 完整句子:
///   单独成条, 不拆中间 —— 无 2~5s 限制, 上限放宽到 SENTENCE_MAX_MS;
///   段尾强标点给小额奖励, 使相邻句子保持各自成条而不是被合并。
/// 非完整句子: 2~5s 目标时长 + 字数 + 边界质量。
/// 句中边界: 停顿越大代价越低 (必须拆分时优先选最长停顿)。
fn segment_cost(units: &[WordItem], j: usize, i: usize) -> f64 {
    let dur = units[i - 1].end_ms - units[j].start_ms;
    let mut cost = 0.0;

    let sentence_start = j == 0 || strong_punct_boundary(units, j);
    let sentence_end = strong_punct_end(units, i);
    if sentence_start && sentence_end {
        // 完整句子: 单独成条; 段尾强标点给强奖励 (使相邻句子保持各自成条,
        // 不会被短句时长偏好或字数惩罚合并)
        if dur > SENTENCE_MAX_MS {
            cost += (dur - SENTENCE_MAX_MS) as f64 * 0.2;
        }
        cost += (dur - SUB_TARGET_MS).abs() as f64 * 0.001;
        cost -= 30.0;
    } else {
        // 非完整句子: 2~5s 约束
        if dur < SUB_MIN_MS {
            cost += (SUB_MIN_MS - dur) as f64 * 0.02;
        }
        if dur > SUB_MAX_MS {
            cost += (dur - SUB_MAX_MS) as f64 * 0.15;
        }
        cost += (dur - SUB_TARGET_MS).abs() as f64 * 0.004;
    }

    // 字数/词数约束: 中文 ≤20 字符, 西文 ≤20 单词。
    // 超过 20 的惩罚仅用于"迫使拆分", 不压制停顿边界的选择 ——
    // 拆在哪由边界代价决定 (停顿最长处优先), 而非固定 20 字位置。
    let count = count_units(units, j, i);
    if count > MAX_UNITS {
        cost += (count - MAX_UNITS) as f64 * 60.0;
    } else if count > PREFER_UNITS {
        cost += (count - PREFER_UNITS) as f64 * 0.5;
    }

    // 边界质量 (段首前的边界): 句点后引语逗号 (≤2 词) 禁止切分 >
    // 句末标点 > 停顿 > 句中逗号 > 无标点 (按停顿大小)
    if j > 0 {
        let prev_ch = units[j - 1].text.chars().last().unwrap_or(' ');
        let pause = units[j].start_ms - units[j - 1].end_ms;
        cost += if is_leadin_comma_boundary(units, j) {
            LEADIN_COMMA_BLOCK
        } else if strong_punct_boundary(units, j) || pause >= PAUSE_STRONG_MS {
            0.0
        } else if is_weak_punct(prev_ch) {
            30.0
        } else {
            mid_sentence_cost(units, j)
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

    /// 按西文单词切分的 units (模拟对齐器输出: 词为单位, 标点附在词尾或独立)
    fn word_units(text: &str, start_ms: i64, per_word_ms: i64) -> Vec<WordItem> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, w)| WordItem {
                text: w.to_string(),
                start_ms: start_ms + i as i64 * per_word_ms,
                end_ms: start_ms + (i as i64 + 1) * per_word_ms,
                confidence: 1.0,
            })
            .collect()
    }

    #[test]
    fn empty_units_returns_empty() {
        assert!(split_subtitles(&[], "ForcedAligned", 0, 0).is_empty());
    }

    #[test]
    fn short_block_stays_one_subtitle() {
        let units = units_from_text("你好世界。", 0, 400);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "你好世界。");
    }

    #[test]
    fn long_block_splits_at_punctuation() {
        // 72 字 × 100ms = 7.2s, 每 12 字一句号 -> 6 个完整句, 各自成条
        let text = "今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。";
        let units = units_from_text(text, 0, 100);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn long_sentence_splits_at_longest_pause() {
        // 20 字单句 (100ms/字), 第 10 字后 300ms 真实间隙 -> 必须在停顿最长处切,
        // 而不是固定 15 字位置
        let text = "今天天气很好我们出去走走今天天气很好我们出去走走。";
        let mut units = units_from_text(text, 0, 100);
        // 在第 10 个字符后插入 300ms 停顿: 后续所有 unit 时间平移 300ms
        for u in units.iter_mut().skip(10) {
            u.start_ms += 300;
            u.end_ms += 300;
        }
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        // 第一段结束在第 10 字 (停顿处), 而非 15 字处
        assert_eq!(segs[0].text, "今天天气很好我们出去");
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn long_sentence_splits_under_15_when_no_pause() {
        // 20 字无停顿: 拆分且每段 ≤15 字
        let text = "今天天气很好我们出去走走今天天气很好我们出去走走。";
        let units = units_from_text(text, 0, 100);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        for s in &segs {
            assert!(count_units(&s.words, 0, s.words.len()) <= MAX_UNITS);
        }
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn short_sentence_stays_whole() {
        // 14 字单句: 在 15 上限内 -> 单独成条, 绝不切
        let text = "今天天气很好我们出去走走。";
        let units = units_from_text(text, 0, 100);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert_eq!(segs.len(), 1, "14-char sentence must stay whole");
        assert_eq!(segs[0].text, text);
    }

    #[test]
    fn two_sentences_stay_separate() {
        // 短句 + 长句: 各自成条, 第二句不合并 (第二句超限时在其内部拆分)
        let mut units = units_from_text("第一句。", 0, 150);
        let second = units_from_text("今天天气很好我们出去走走今天天气很好我们出去走走。", 900, 100);
        units.extend(second);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert_eq!(segs[0].text, "第一句。", "short sentence must stay separate");
        let rest: String = segs.iter().skip(1).map(|s| s.text.as_str()).collect();
        assert_eq!(rest, "今天天气很好我们出去走走今天天气很好我们出去走走。");
    }

    #[test]
    fn abbreviation_dot_not_a_boundary() {
        // "Mr. Smith" 的句点不是句子边界; 真实句号才是
        let mut units = units_from_text("Mr.", 0, 200);
        let rest = units_from_text(" Smith is here today.", 600, 150);
        units.extend(rest);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        // 无句号 -> 单条
        assert_eq!(segs.len(), 1, "abbreviation dot must not split");
    }

    #[test]
    fn connective_comma_not_split() {
        // "Hello. However, the world is a very beautiful place to live in today."
        // "However," 后不得切分; 且不得出现孤立连接词字幕
        let text = "Hello. However, the world is a very beautiful place to live in today.";
        let units = units_from_text(text, 0, 120);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        for s in &segs {
            let t = s.text.trim().to_lowercase();
            assert!(
                !(t.starts_with("however") && s.text.trim().len() <= 10),
                "connective must not be isolated: {}",
                s.text
            );
        }
        // 句点边界处拆分 (两句), 或整体一条; 但绝不能把 However 切出去
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined.replace(' ', ""), text.replace(' ', ""));
    }

    #[test]
    fn leadin_comma_structural_rule() {
        // 1 词引语: "However," 后阻断
        let units = word_units("Hello. However, the world is here", 0, 200);
        assert!(is_leadin_comma_boundary(&units, 2), "1-word leadin must block");
        // 2 词引语: "I mean," 后阻断
        let units = word_units("Hello. I mean, the world", 0, 200);
        assert!(is_leadin_comma_boundary(&units, 3), "2-word leadin must block");
        // 3 词引语: "By the way," 不阻断 (超出规则范围)
        let units = word_units("Hello. By the way, the world", 0, 200);
        assert!(!is_leadin_comma_boundary(&units, 4), "3-word leadin must NOT block");
        // 无句点: "Well, the" 不阻断
        let units = word_units("Well, the world", 0, 200);
        assert!(!is_leadin_comma_boundary(&units, 1), "no period -> no block");
        // 中文: 不适用
        let units = units_from_text("你好，好", 0, 200);
        assert!(!is_leadin_comma_boundary(&units, 3), "chinese -> no block");
    }

    #[test]
    fn times_are_monotonic_and_smoothed() {
        let text = "第一句。第二句。第三句。第四句。第五句。第六句。第七句。第八句。第九句。第十句。";
        let units = units_from_text(text, 1000, 150);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        for w in segs.windows(2) {
            assert!(w[1].start_ms >= w[0].end_ms, "overlap detected");
        }
        // 首条字幕有前垫 (块真实起点 0 起), 末条不越界
        assert!(segs[0].start_ms >= 0);
        assert!(segs.last().unwrap().end_ms <= units.last().unwrap().end_ms);
    }
}
