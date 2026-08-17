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
/// 字幕长度硬上限: 中文 ≤25 字符, 西文 ≤25 单词 (标点不计)
const MAX_UNITS: usize = 25;
/// 单条偏好的长度 (低于该值无长度代价)
const PREFER_UNITS: usize = 12;
/// 开始时间轻微视觉平滑 (ms)
const START_PAD_MS: i64 = 150;
/// 结束时间轻微视觉平滑 (ms)
const END_PAD_MS: i64 = 200;
/// 词间停顿超过该值视为强边界 (ms)
const PAUSE_STRONG_MS: i64 = 250;
/// 句中无停顿拆分的基准代价 (必须拆时优先选最长停顿)
const MID_SENTENCE_BASE: f64 = 1000.0;
/// 句点后引语逗号 (≤2 词后接逗号) 禁止切分的代价
const LEADIN_COMMA_BLOCK: f64 = 500.0;
/// 弱标点 (逗号等) 边界代价: 必须远低于句中无标点边界的最小代价 (500.0),
/// 保证"有标点优先于无标点停顿"的语言学优先级
const WEAK_PUNCT_COST: f64 = 10.0;
/// 单条字幕时长硬上限 (ms): DP 剪枝用, 超过此长度的候选段不再计算。
/// 时间单调时内层循环可提前 break, 长块 (N≈700) 下 DP 近似 O(N)。
const MAX_SEG_MS: i64 = 15000;

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

/// 移除文本中所有的全角句号 "。"
pub fn remove_periods(text: &str) -> String {
    text.replace('。', "")
}


/// 缩写保护: 短字母词后的句点 ("Mr." "Dr." "St." "etc." "U.S.") 不是句子边界;
/// 数字小数点 ("3.5") 同理。返回 true 表示该句点应视为普通句点 (强边界)。
fn is_real_period(units: &[WordItem], dot_idx: usize) -> bool {
    let dot_text = &units[dot_idx].text;
    let core = dot_text.trim_end_matches('.');
    if core.chars().any(|c| c.is_alphanumeric()) {
        // 附着格式: 句点与词同 unit, 直接检查该词本身
        return classify_period_word(core);
    }
    // 独立/逐字符格式: 向前合并连续 ASCII 字母数字 unit, 还原完整单词再判定。
    // 注意必须用 is_ascii_alphanumeric: CJK 字符属于 Unicode Alphabetic,
    // 用 is_alphanumeric 会把中文一并并进"单词" 。
    // 遇到空格/标点/CJK unit 立即停止
    let mut word = String::new();
    let mut k = dot_idx;
    while k > 0 {
        k -= 1;
        let t = units[k].text.trim();
        if t.is_empty() || !t.chars().all(|c| c.is_ascii_alphanumeric()) {
            break;
        }
        word.insert_str(0, t);
    }
    classify_period_word(&word)
}

/// 判定句点前的"单词"是缩写/数字 (返回 false) 还是真实句末 (返回 true)。
fn classify_period_word(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    if chars.is_empty() {
        return true;
    }
    // 数字: 小数点
    if chars.iter().all(|c| c.is_ascii_digit()) {
        return false;
    }
    // 短字母缩写: Mr. Dr. St. No. etc. (≤4 个字母)
    if chars.iter().all(|c| c.is_ascii_alphabetic()) && chars.len() <= 4 {
        return false;
    }
    true
}

/// 边界 (unit j 之前) 是否为强标点边界。句点带缩写/小数点保护。
fn strong_punct_boundary(units: &[WordItem], j: usize) -> bool {
    if j == 0 {
        return false;
    }
    let ch = units[j - 1].text.trim_end().chars().last().unwrap_or(' ');
    if ch == '.' {
        return is_real_period(units, j - 1);
    }
    is_strong_punct(ch)
}

/// 边界 (unit j 之前) 是否为标点边界 (强标点或逗号等弱标点)。
fn is_punct_boundary(units: &[WordItem], j: usize) -> bool {
    if j == 0 {
        return true;
    }
    let ch = units[j - 1].text.trim_end().chars().last().unwrap_or(' ');
    if ch == '.' {
        return is_real_period(units, j - 1);
    }
    is_strong_punct(ch) || is_weak_punct(ch)
}

/// 段尾 (units[i-1]) 是否为强标点。
fn strong_punct_end(units: &[WordItem], i: usize) -> bool {
    if i == 0 {
        return false;
    }
    let ch = units[i - 1].text.trim_end().chars().last().unwrap_or(' ');
    if ch == '.' {
        return is_real_period(units, i - 1);
    }
    is_strong_punct(ch)
}

/// 段尾 (units[i-1]) 是否为弱标点 (逗号、分号等)。
fn weak_punct_end(units: &[WordItem], i: usize) -> bool {
    if i == 0 {
        return false;
    }
    let ch = units[i - 1].text.trim_end().chars().last().unwrap_or(' ');
    is_weak_punct(ch)
}

/// 句中边界的代价: 停顿越大代价越低, 必须拆分时优先选最长停顿处
fn mid_sentence_cost(units: &[WordItem], j: usize) -> f64 {
    let pause = units[j]
        .start_ms
        .saturating_sub(units[j - 1].end_ms)
        .min(100) as f64;
    (MID_SENTENCE_BASE - pause).max(500.0)
}

/// 句点后引语逗号边界: 句点后、逗号前的西文单词 ≤2 个时忽略该逗号不切
/// ("However," "Yeah," "I mean," 等话语标记后不切; 结构化规则, 无需白名单)。
/// 语音块开头视为天然句首 (前一句的句点被块边界截断), 段首引语逗号同样受保护。
fn is_leadin_comma_boundary(units: &[WordItem], j: usize) -> bool {
    if j == 0 {
        return false;
    }
    let prev_ch = units[j - 1].text.trim_end().chars().last().unwrap_or(' ');
    if !is_comma_punct(prev_ch) {
        return false;
    }

    let mut raw = String::new();
    for (k, u) in units[0..j].iter().enumerate() {
        if k > 0
            && !raw.ends_with(char::is_whitespace)
            && !u.text.starts_with(char::is_whitespace)
            && !is_word_internal_boundary(units, k)
        {
            raw.push(' ');
        }
        raw.push_str(&u.text);
    }

    let text_up_to_comma = raw.trim_end_matches(|c: char| is_comma_punct(c) || c.is_whitespace());
    let sentence_start = match text_up_to_comma.rfind(|c: char| is_strong_punct(c) || c == '.') {
        Some(pos) => &text_up_to_comma[pos + 1..],
        None => text_up_to_comma,
    };

    if sentence_start.chars().any(is_cjk_char) {
        return false;
    }

    let words: Vec<&str> = sentence_start
        .split(|c: char| c.is_whitespace() || is_weak_punct(c))
        .filter(|s| s.chars().any(|c| c.is_alphanumeric()))
        .collect();

    !words.is_empty() && words.len() <= 2
}

/// 统计一段的字数: 中文按字符数, 西文按真实单词数 (纯标点/空格不计)
fn count_units(units: &[WordItem], j: usize, i: usize) -> usize {
    let mut count = 0usize;
    let mut in_western_word = false;
    for (k, u) in units[j..i].iter().enumerate() {
        let global_idx = j + k;
        for ch in u.text.chars() {
            if is_cjk_char(ch) {
                if in_western_word {
                    count += 1;
                    in_western_word = false;
                }
                if !is_strong_punct(ch) && !is_weak_punct(ch) && !ch.is_whitespace() {
                    count += 1;
                }
            } else if is_western_word_char(ch) {
                in_western_word = true;
            } else {
                if in_western_word {
                    count += 1;
                    in_western_word = false;
                }
            }
        }
        if in_western_word && (global_idx + 1 == i || !is_word_internal_boundary(units, global_idx + 1)) {
            count += 1;
            in_western_word = false;
        }
    }
    count
}

/// 将一段带标点文本及其词/字级时间戳拆成 2~5 秒的字幕 (动态规划)。
///
/// - `units` 为空时返回空 Vec, 调用方回退为整段一条字幕;
/// - 拆分点优先落在句末标点/长停顿, 其次句中逗号, 无标点按时长硬切;
/// - 输出字幕的时间已做轻微视觉平滑且保证相邻不重叠:
///   平滑重叠时 (上一句尾垫吞掉本句头垫) 在两句真实时间边界之间取中点,
///   上一句回缩尾垫, 保证本句字幕不晚于其语音起点出现;
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

    // dp[i] = 把 units[0..i) 拆成若干字幕的最小代价; prev[i] = 最后一段的起点。
    // 内层 j 从大到小扫描: 时间单调时 j 越小段越长 (dur 越大),
    // 超过 MAX_SEG_MS 直接剪枝, 长块下 DP 近似 O(N) 而非 O(N^2)。
    let mut dp = vec![f64::INFINITY; n + 1];
    let mut prev = vec![0usize; n + 1];
    dp[0] = 0.0;

    for i in 1..=n {
        for j in (0..i).rev() {
            let dur = units[i - 1].end_ms - units[j].start_ms;
            if dur > MAX_SEG_MS {
                break;
            }
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
    let mut out: Vec<TranscriptionSegment> = Vec::with_capacity(bounds.len() - 1);
    let mut prev_end = block_start;
    let mut prev_real_end = block_start;
    for w in bounds.windows(2) {
        let (j, i) = (w[0], w[1]);
        if j >= i {
            continue;
        }
        let raw_text: String = units[j..i].iter().map(|u| u.text.as_str()).collect();
        let text = remove_periods(&raw_text);
        let real_start = units[j].start_ms;
        let real_end = units[i - 1].end_ms;
        let mut start = real_start - START_PAD_MS;
        let mut end = real_end + END_PAD_MS;
        if start < block_start {
            start = block_start;
        }
        if end > block_end {
            end = block_end;
        }
        if start < prev_end {
            // 平滑重叠: 在两句真实时间边界之间取中点,
            // 上一句回缩尾垫, 本句保留部分头垫 —— 避免本句字幕
            // 被推迟到语音起点之后 (最多晚 ~200ms) 才显示
            let mid = (prev_real_end + real_start) / 2;
            if mid < prev_end {
                if let Some(last) = out.last_mut() {
                    last.end_ms = mid.max(last.start_ms);
                    prev_end = last.end_ms;
                }
            }
            start = mid;
            if start < prev_end {
                // 保底 (时间逆序等异常对齐数据): 绝不与前一条重叠
                start = prev_end;
            }
        }
        if end <= start {
            end = start + 1;
        }
        prev_end = end;
        prev_real_end = real_end;
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

/// 判断字符是否属于西文单词构成字符 (字母、数字、撇号、连字符等)
fn is_western_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '\'' || c == '-' || c == '’'
}

/// 判断边界 (units[j] 之前) 是否处于西文/英文单词内部。
/// 若前一个 unit 末尾是西文单词字符，且当前 unit 开头也是西文单词字符（中间没有空格或标点），
/// 则说明此处位于同一个单词内部，无论任何时候绝对禁止切分！
fn is_word_internal_boundary(units: &[WordItem], j: usize) -> bool {
    if j == 0 || j >= units.len() {
        return false;
    }
    let prev_text = &units[j - 1].text;
    let next_text = &units[j].text;
    let prev_ch = match prev_text.chars().last() {
        Some(c) => c,
        None => return false,
    };
    let next_ch = match next_text.chars().next() {
        Some(c) => c,
        None => return false,
    };

    // 前后均为西文单词字符 -> 处于单词内部
    is_western_word_char(prev_ch) && is_western_word_char(next_ch)
}

/// 边界 (units[j] 之前) 的切分代价, 优先级 (由低到高):
/// 句末标点/长停顿 (0) > 弱标点逗号 (WEAK_PUNCT_COST) > 句中无标点 (按停顿大小);
/// 句点后引语逗号 (≤2 词) 禁止切分 (LEADIN_COMMA_BLOCK);
/// 单词内部绝对禁止切分 (INFINITY)。
fn boundary_cost(units: &[WordItem], j: usize) -> f64 {
    if j == 0 {
        return 0.0;
    }
    // 无论任何时候，绝对禁止从英文/西文单词内部断开
    if is_word_internal_boundary(units, j) {
        return f64::INFINITY;
    }
    let prev_ch = units[j - 1].text.chars().last().unwrap_or(' ');
    let pause = units[j].start_ms - units[j - 1].end_ms;
    if is_leadin_comma_boundary(units, j) {
        LEADIN_COMMA_BLOCK
    } else if strong_punct_boundary(units, j) || pause >= PAUSE_STRONG_MS {
        0.0
    } else if is_weak_punct(prev_ch) {
        WEAK_PUNCT_COST
    } else {
        mid_sentence_cost(units, j)
    }
}

/// 一段字幕 (units[j..i)) 的拆分代价。
///
/// 完整句子:
///   单独成条, 不拆中间 —— 无 2~5s 限制, 上限放宽到 SENTENCE_MAX_MS;
///   段尾强标点给小额奖励, 使相邻句子保持各自成条而不是被合并。
/// 完整分句 (逗号/分号等天然分句):
///   优先在逗号处拆分，放宽到 8000ms。
/// 句中无标点边界:
///   严格 2~5s 约束 + 高昂的切词惩罚。
fn segment_cost(units: &[WordItem], j: usize, i: usize) -> f64 {
    let b_cost = boundary_cost(units, j);
    if b_cost.is_infinite() {
        return f64::INFINITY;
    }

    let dur = units[i - 1].end_ms - units[j].start_ms;
    let mut cost = 0.0;

    let sentence_start = j == 0 || strong_punct_boundary(units, j);
    let sentence_end = strong_punct_end(units, i);
    let clause_start = j == 0 || is_punct_boundary(units, j);
    let clause_end = strong_punct_end(units, i) || weak_punct_end(units, i);

    if sentence_start && sentence_end {
        // 完整句子: 单独成条; 段尾强标点给强奖励 (使相邻句子保持各自成条,
        // 不会被短句时长偏好或字数惩罚合并)
        if dur > SENTENCE_MAX_MS {
            cost += (dur - SENTENCE_MAX_MS) as f64 * 0.2;
        }
        cost += (dur - SUB_TARGET_MS).abs() as f64 * 0.001;
        cost -= 30.0;
    } else if clause_start && clause_end {
        // 完整分句 (逗号/分号/句号边界完整): 允许保持分句完整，上限放宽到 8000ms
        if dur > 8000 {
            cost += (dur - 8000) as f64 * 0.15;
        }
        cost += (dur - SUB_TARGET_MS).abs() as f64 * 0.002;
    } else {
        // 句中无标点硬切: 严格 2~5s 约束
        if dur < SUB_MIN_MS {
            cost += (SUB_MIN_MS - dur) as f64 * 0.02;
        }
        if dur > SUB_MAX_MS {
            cost += (dur - SUB_MAX_MS) as f64 * 0.15;
        }
        cost += (dur - SUB_TARGET_MS).abs() as f64 * 0.004;
    }

    // 字数/词数约束: 中文 ≤MAX_UNITS 字符, 西文 ≤MAX_UNITS 单词。
    // 超过 MAX_UNITS 的惩罚仅用于"迫使拆分", 不压制停顿边界的选择
    // 拆在哪由边界代价决定 (停顿最长处优先), 而非固定字数位置。
    let count = count_units(units, j, i);
    if count > MAX_UNITS {
        cost += (count - MAX_UNITS) as f64 * 60.0;
    } else if count > PREFER_UNITS {
        cost += (count - PREFER_UNITS) as f64 * 0.5;
    }

    cost += b_cost;
    cost
}

// ================= 以下测试 =======================

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

    /// 按西文单词切分的 units (模拟对齐器输出: 词为单位, 标点附在词尾或独立, 词间含空格)
    fn word_units(text: &str, start_ms: i64, per_word_ms: i64) -> Vec<WordItem> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, w)| WordItem {
                text: format!("{} ", w),
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
    fn removes_full_stop_periods() {
        assert_eq!(remove_periods("你好。世界。"), "你好世界");
        assert_eq!(remove_periods("这是。一个。测试。"), "这是一个测试");
        assert_eq!(remove_periods("Hello. World!"), "Hello. World!");
    }

    #[test]
    fn short_block_stays_one_subtitle() {
        let units = units_from_text("你好世界。", 0, 400);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "你好世界");
    }

    #[test]
    fn long_block_splits_at_punctuation() {
        // 72 字 × 100ms = 7.2s, 每 12 字一句号 -> 6 个完整句, 各自成条
        let text = "今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。今天天气很好我们出去走走。";
        let units = units_from_text(text, 0, 100);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text.replace('。', ""));
    }

    #[test]
    fn long_sentence_splits_at_longest_pause() {
        // 25 字单句 (100ms/字), 第 10 字后 300ms 真实间隙 -> 必须在停顿最长处切
        let text = "今天天气很好我们出去走走今天天气很好我们出去走走很有趣。";
        let mut units = units_from_text(text, 0, 100);
        // 在第 10 个字符后插入 300ms 停顿: 后续所有 unit 时间平移 300ms
        for u in units.iter_mut().skip(10) {
            u.start_ms += 300;
            u.end_ms += 300;
        }
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        assert_eq!(segs[0].text, "今天天气很好我们出去");
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text.replace('。', ""));
    }

    #[test]
    fn long_sentence_splits_under_max_units_when_no_pause() {
        // 48 字无句尾标点无停顿: 强制拆分且每段 ≤MAX_UNITS
        let text = "今天天气很好我们出去走走今天天气很好我们出去走走今天天气很好我们出去走走今天天气很好我们出去走走";
        let units = units_from_text(text, 0, 100);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert!(segs.len() >= 2, "expected >=2 subtitles, got {}", segs.len());
        for s in &segs {
            assert!(count_units(&s.words, 0, s.words.len()) <= MAX_UNITS);
        }
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text.replace('。', ""));
    }

    #[test]
    fn short_sentence_stays_whole() {
        // 14 字单句: 在 20 上限内 -> 单独成条, 绝不切
        let text = "今天天气很好我们出去走走。";
        let units = units_from_text(text, 0, 100);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert_eq!(segs.len(), 1, "14-char sentence must stay whole");
        assert_eq!(segs[0].text, "今天天气很好我们出去走走");
    }

    #[test]
    fn two_sentences_stay_separate() {
        // 短句 + 长句: 各自成条, 第二句不合并 (第二句超限时在其内部拆分)
        let mut units = units_from_text("第一句。", 0, 150);
        let second = units_from_text("今天天气很好我们出去走走今天天气很好我们出去走走。", 900, 100);
        units.extend(second);
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert_eq!(segs[0].text, "第一句", "short sentence must stay separate");
        let rest: String = segs.iter().skip(1).map(|s| s.text.as_str()).collect();
        assert_eq!(rest, "今天天气很好我们出去走走今天天气很好我们出去走走");
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
        // 块首引语 (块起点即句首, 前一句句点被块边界截断): "Well," 同样阻断
        let units = word_units("Well, the world", 0, 200);
        assert!(is_leadin_comma_boundary(&units, 1), "block-start leadin must block");
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

    #[test]
    fn comma_splits_instead_of_breaking_words() {
        // 用户真实案例: 30 字符长句 (~9.6s), 必须在逗号处断句, 严禁在“热烈之中”等词语中间断开
        let text = "而一池荷花教给我们的是，热烈之中仍可怀有一寸自己的月白风清。";
        let units = units_from_text(text, 0, 320); // 30 字 × 320ms = 9.6s
        let segs = split_subtitles(&units, "ForcedAligned", 0, units.last().unwrap().end_ms);
        assert_eq!(segs.len(), 2, "9.6s sentence should split into 2 subtitles at comma");
        assert_eq!(segs[0].text, "而一池荷花教给我们的是，");
        assert_eq!(segs[1].text, "热烈之中仍可怀有一寸自己的月白风清");
    }
}
