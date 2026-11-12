use crate::{invalid, Error, Result};
use serde_json::Value;

#[derive(Clone, Debug)]
pub(crate) struct BufferSpec {
    pub leb128: bool,
    blocked: bool,
    varint_frames: bool,
    compression: Compression,
    block_size: usize,
}

#[derive(Clone, Debug)]
enum Compression {
    None,
    Lz4,
    Lz4Sized,
    Zstd,
    ZstdSized,
}

impl BufferSpec {
    pub fn frame_size(&self, header: &[u8]) -> Result<usize> {
        if !self.blocked {
            return Err(Error::Unsupported(
                "indexed reads of unblocked streams".into(),
            ));
        }
        let mut cursor = Cursor::new(header, false);
        let n = if self.varint_frames {
            cursor.varint(32)? as usize
        } else {
            cursor.u32()? as usize
        };
        if n > self.block_size * 2 + 64 {
            return Err(invalid("frame size exceeds buffer bound"));
        }
        Ok(cursor.position + n)
    }

    pub fn matrix() -> Self {
        Self {
            leb128: false,
            blocked: true,
            varint_frames: false,
            compression: Compression::Lz4,
            block_size: 32768,
        }
    }

    pub fn parse(spec: &Value) -> Result<Self> {
        let mut result = Self {
            leb128: false,
            blocked: false,
            varint_frames: false,
            compression: Compression::None,
            block_size: 65536,
        };
        let mut current = spec;
        loop {
            let name = current["name"]
                .as_str()
                .ok_or_else(|| invalid("buffer spec lacks name"))?;
            match name {
                "LEB128BufferSpec" => result.leb128 = true,
                "BlockingBufferSpec" => {
                    result.blocked = true;
                    result.block_size = current["blockSize"]
                        .as_u64()
                        .ok_or_else(|| invalid("blockSize"))?
                        as usize;
                    if result.block_size == 0 || result.block_size > 65536 {
                        return Err(invalid("block size outside 1..65536"));
                    }
                }
                "LZ4BlockBufferSpec" | "LZ4FastBlockBufferSpec" | "LZ4HCBlockBufferSpec" => {
                    result.compression = Compression::Lz4
                }
                "LZ4SizeBasedBlockBufferSpec" => result.compression = Compression::Lz4Sized,
                "ZstdBlockBufferSpec" => result.compression = Compression::Zstd,
                "ZstdSizedBasedBlockBufferSpec" => result.compression = Compression::ZstdSized,
                "StreamBlockBufferSpec" | "StreamBufferSpec" => break,
                "StreamBlockBufferSpec2" => {
                    result.varint_frames = true;
                    break;
                }
                other => return Err(Error::Unsupported(format!("buffer {other}"))),
            }
            current = &current["child"];
        }
        Ok(result)
    }

    /// Returns one decompressed frame and the physical length consumed.
    pub fn frame(&self, input: &[u8]) -> Result<(Vec<u8>, usize)> {
        let mut cursor = Cursor::new(input, false);
        let length = if self.varint_frames {
            cursor.varint(32)? as usize
        } else {
            cursor.u32()? as usize
        };
        if length > self.block_size * 2 + 64 {
            return Err(invalid("compressed frame exceeds buffer bound"));
        }
        let payload = cursor.take(length)?;
        let consumed = cursor.position;
        let mut content = Cursor::new(payload, false);
        let (size, data, compression) = match &self.compression {
            Compression::None => (length, payload, &Compression::None),
            Compression::Lz4 | Compression::Zstd => {
                let size = content.u32()? as usize;
                (size, &payload[4..], &self.compression)
            }
            Compression::Lz4Sized => match content.u32()? {
                0 => (length - 4, &payload[4..], &Compression::None),
                1 => {
                    let size = content.u32()? as usize;
                    (size, &payload[8..], &Compression::Lz4)
                }
                _ => return Err(invalid("invalid LZ4 compression flag")),
            },
            Compression::ZstdSized => {
                let size = content.u32()? as usize;
                if size & 1 == 0 {
                    (length - 4, &payload[4..], &Compression::None)
                } else {
                    (size >> 1, &payload[4..], &Compression::Zstd)
                }
            }
        };
        if size > self.block_size {
            return Err(invalid("decompressed frame exceeds blockSize"));
        }
        let decoded = match compression {
            Compression::None => data.to_vec(),
            Compression::Lz4 => {
                lz4_flex::block::decompress(data, size).map_err(|e| invalid(format!("LZ4: {e}")))?
            }
            Compression::Zstd => zstd::bulk::decompress(data, size)?,
            _ => unreachable!(),
        };
        if decoded.len() != size {
            return Err(invalid("decompressed frame length mismatch"));
        }
        Ok((decoded, consumed))
    }

    pub fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.blocked {
            return Ok(data.to_vec());
        }
        let mut output = Vec::new();
        let mut offset = 0;
        while offset < data.len() {
            let (block, consumed) = self.frame(&data[offset..])?;
            output.extend(block);
            offset += consumed;
        }
        Ok(output)
    }
}

pub(crate) struct Cursor<'a> {
    input: &'a [u8],
    pub position: usize,
    pub leb128: bool,
}

impl<'a> Cursor<'a> {
    pub fn new(input: &'a [u8], leb128: bool) -> Self {
        Self {
            input,
            position: 0,
            leb128,
        }
    }
    pub fn remaining(&self) -> usize {
        self.input.len() - self.position
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(n)
            .ok_or_else(|| invalid("length overflow"))?;
        let bytes = self.input.get(self.position..end).ok_or(Error::Truncated {
            position: self.position,
            needed: n,
        })?;
        self.position = end;
        Ok(bytes)
    }
    pub fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn varint(&mut self, bits: u32) -> Result<u64> {
        let mut result = 0;
        for shift in (0..bits).step_by(7) {
            let byte = self.byte()?;
            if shift + 7 >= bits && (byte >> (bits - shift)) != 0 {
                return Err(invalid("varint overflow"));
            }
            result |= ((byte & 127) as u64) << shift;
            if byte & 128 == 0 {
                return Ok(result);
            }
        }
        Err(invalid("unterminated varint"))
    }
    pub fn int(&mut self) -> Result<i32> {
        Ok(if self.leb128 {
            self.varint(32)? as i32
        } else {
            self.u32()? as i32
        })
    }
    pub fn long(&mut self) -> Result<i64> {
        Ok(if self.leb128 {
            self.varint(64)? as i64
        } else {
            self.u64()? as i64
        })
    }
    pub fn length(&mut self, varint: bool) -> Result<usize> {
        let n = if varint {
            self.varint(32)? as i32
        } else {
            self.int()?
        };
        if !(0..=16_777_216).contains(&n) {
            return Err(invalid(format!("invalid collection length {n}")));
        }
        Ok(n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn codecs_and_frame_versions() {
        let input = b"Hail native format: repeated repeated repeated";
        for (name, payload) in [
            ("LZ4FastBlockBufferSpec", lz4_flex::block::compress(input)),
            (
                "ZstdBlockBufferSpec",
                zstd::bulk::compress(input, 1).unwrap(),
            ),
        ] {
            for version in ["StreamBlockBufferSpec", "StreamBlockBufferSpec2"] {
                let spec = BufferSpec::parse(&json!({
                    "name":"BlockingBufferSpec", "blockSize":65536,
                    "child":{"name":name, "blockSize":65536, "child":{"name":version}}
                }))
                .unwrap();
                let n = payload.len() + 4;
                let mut data = if version.ends_with('2') {
                    vec![n as u8]
                } else {
                    (n as u32).to_le_bytes().to_vec()
                };
                data.extend((input.len() as u32).to_le_bytes());
                data.extend(&payload);
                assert_eq!(spec.decode(&data).unwrap(), input);
                assert!(spec.decode(&data[..data.len() - 1]).is_err());
            }
        }
    }

    #[test]
    fn sized_compression_raw_and_compressed() {
        let input = b"abcabcabcabc";
        for name in [
            "LZ4SizeBasedBlockBufferSpec",
            "ZstdSizedBasedBlockBufferSpec",
        ] {
            let spec = BufferSpec::parse(&json!({
                "name":"BlockingBufferSpec","blockSize":32768,
                "child":{"name":name,"child":{"name":"StreamBlockBufferSpec"}}
            }))
            .unwrap();
            for compressed in [false, true] {
                let mut payload = Vec::new();
                if name.starts_with("LZ4") {
                    payload.extend(u32::from(compressed).to_le_bytes());
                    if compressed {
                        payload.extend((input.len() as u32).to_le_bytes());
                        payload.extend(lz4_flex::block::compress(input));
                    } else {
                        payload.extend(input);
                    }
                } else {
                    payload.extend(if compressed {
                        (input.len() as u32 * 2 + 1).to_le_bytes()
                    } else {
                        0u32.to_le_bytes()
                    });
                    if compressed {
                        payload.extend(zstd::bulk::compress(input, 1).unwrap());
                    } else {
                        payload.extend(input);
                    }
                }
                let mut frame = (payload.len() as u32).to_le_bytes().to_vec();
                frame.extend(payload);
                assert_eq!(spec.decode(&frame).unwrap(), input);
            }
        }
    }

    #[test]
    fn overflow_and_unterminated_varints() {
        assert!(Cursor::new(&[255; 10], false).varint(64).is_err());
        assert!(Cursor::new(&[128; 5], false).varint(32).is_err());
        assert_eq!(
            Cursor::new(&[255, 255, 255, 255, 15], true).int().unwrap(),
            -1
        );
    }
}
