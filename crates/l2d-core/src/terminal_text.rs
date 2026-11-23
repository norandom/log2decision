//! Character handling shared by local terminal clients.

/// Whether a character changes how a terminal lays out or interprets what follows
/// without being visible itself: control characters (C0, C1), and the format
/// characters that hide text or reverse its direction (soft hyphen, the Arabic letter
/// mark, zero-width characters and direction marks U+200B to U+200F, line and paragraph
/// separators and the bidirectional embeddings and overrides U+2028 to U+202E, the word
/// joiner, invisible operators and bidirectional isolates U+2060 to U+206F, the byte
/// order mark, the interlinear annotation marks and the tag characters).
pub fn is_invisible_or_control(character: char) -> bool {
    character.is_control()
        || matches!(
            character,
            '\u{ad}'
                | '\u{61c}'
                | '\u{180e}'
                | '\u{200b}'..='\u{200f}'
                | '\u{2028}'..='\u{202e}'
                | '\u{2060}'..='\u{206f}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
                | '\u{e0000}'..='\u{e007f}'
        )
}
