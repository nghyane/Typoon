/// Extract the first numeric sequence from a directory name as chapter number.
pub fn parse_chapter_number(name: &str) -> Option<usize> {
    name.chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}
