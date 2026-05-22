use std::fmt;

pub const MAGIC: [u8; 4] = [b'M', b'C', b'Z', 0x01];
pub const RIFF: [u8; 4] = [b'R', b'I', b'F', b'F'];
pub const MCZD: [u8; 4] = [b'M', b'C', b'Z', b'd'];
pub const VERSION: u8 = 1;
pub const HEADER_SIZE: usize = 8;
pub const INDEX_ENTRY_SIZE: usize = 16;

/// VP8L 1×1 white image data (17 bytes). Dims patched at bytes [1..5].
pub const VP8L_DATA: [u8; 17] = [
    0x2f,
    0x00,0x00,0x00,0x00,
    0x07,0xd0,0xff,0xfe,0xf7,0xbf,0xff,0x81,0x88,0xe8,0x7f,0x00,
];

/// RIFF overhead before MCZ data: RIFF(12) + VP8L chunk(8+17+1pad) + MCZd header(8) = 46
pub const COVER_PREFIX: usize = 46;

/// Scan RIFF chunks to find MCZd, return offset where MCZ data starts.
pub fn mcz_offset(data: &[u8]) -> Result<usize, ParseError> {
    if data.len() < 8 {
        return Err(ParseError::TooShort);
    }
    if data[0..4] == MAGIC {
        return Ok(0);
    }
    if data[0..4] != RIFF {
        return Err(ParseError::BadMagic);
    }
    let riff_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let end = 8 + riff_size;
    let mut pos = 12; // skip RIFF header + "WEBP"
    while pos + 8 <= end && pos + 8 <= data.len() {
        if data[pos..pos + 4] == MCZD {
            return Ok(pos + 8);
        }
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        pos += 8 + size + (size % 2);
    }
    Err(ParseError::BadMagic)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ImageFormat {
    WebP = 0,
    Jpeg = 1,
    Jxl = 2,
}

impl ImageFormat {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::WebP),
            1 => Some(Self::Jpeg),
            2 => Some(Self::Jxl),
            _ => None,
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::WebP => "webp",
            Self::Jpeg => "jpg",
            Self::Jxl => "jxl",
        }
    }

    pub fn mime(&self) -> &'static str {
        match self {
            Self::WebP => "image/webp",
            Self::Jpeg => "image/jpeg",
            Self::Jxl => "image/jxl",
        }
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WebP => write!(f, "WebP"),
            Self::Jpeg => write!(f, "JPEG"),
            Self::Jxl => write!(f, "JXL"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PageInfo {
    pub index: u16,
    pub offset: u32,
    pub size: u32,
    pub width: u16,
    pub height: u16,
    pub format: ImageFormat,
}

#[derive(Debug, Clone)]
pub struct MCZIndex {
    pub version: u8,
    pub pages: Vec<PageInfo>,
}

impl MCZIndex {
    pub fn data_offset(page_count: u16) -> usize {
        HEADER_SIZE + page_count as usize * INDEX_ENTRY_SIZE
    }
}

// ── Serialize ───────────────────────────────────────────────────────

pub fn write_header(buf: &mut Vec<u8>, page_count: u16) {
    buf.extend_from_slice(&MAGIC);
    buf.push(VERSION);
    buf.push(0); // flags
    buf.extend_from_slice(&page_count.to_le_bytes());
}

pub fn write_index_entry(buf: &mut Vec<u8>, page: &PageInfo) {
    buf.extend_from_slice(&page.offset.to_le_bytes());
    buf.extend_from_slice(&page.size.to_le_bytes());
    buf.extend_from_slice(&page.width.to_le_bytes());
    buf.extend_from_slice(&page.height.to_le_bytes());
    buf.push(page.format as u8);
    buf.extend_from_slice(&[0u8; 3]); // reserved
}

// ── Parse ───────────────────────────────────────────────────────────

pub fn parse_header(data: &[u8], off: usize) -> Result<(u8, u16), ParseError> {
    if data.len() < off + HEADER_SIZE {
        return Err(ParseError::TooShort);
    }
    if data[off..off + 4] != MAGIC {
        return Err(ParseError::BadMagic);
    }
    let version = data[off + 4];
    let page_count = u16::from_le_bytes([data[off + 6], data[off + 7]]);
    Ok((version, page_count))
}

pub fn parse_index(data: &[u8], off: usize, page_count: u16) -> Result<Vec<PageInfo>, ParseError> {
    let needed = off + HEADER_SIZE + page_count as usize * INDEX_ENTRY_SIZE;
    if data.len() < needed {
        return Err(ParseError::TooShort);
    }

    let mut pages = Vec::with_capacity(page_count as usize);
    for i in 0..page_count {
        let base = off + HEADER_SIZE + i as usize * INDEX_ENTRY_SIZE;
        let d = &data[base..base + INDEX_ENTRY_SIZE];

        let format = ImageFormat::from_u8(d[12]).ok_or(ParseError::BadFormat(d[12]))?;
        pages.push(PageInfo {
            index: i,
            offset: u32::from_le_bytes([d[0], d[1], d[2], d[3]]),
            size: u32::from_le_bytes([d[4], d[5], d[6], d[7]]),
            width: u16::from_le_bytes([d[8], d[9]]),
            height: u16::from_le_bytes([d[10], d[11]]),
            format,
        });
    }
    Ok(pages)
}

// ── Errors ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ParseError {
    TooShort,
    BadMagic,
    BadFormat(u8),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "data too short"),
            Self::BadMagic => write!(f, "invalid MCZ magic bytes"),
            Self::BadFormat(v) => write!(f, "unknown image format: {v}"),
        }
    }
}

impl std::error::Error for ParseError {}
