use crate::{buffer::Cursor, invalid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HailType {
    pub name: String,
    pub required: bool,
    pub fields: Vec<(String, HailType)>,
    pub args: Vec<HailType>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Binary(Vec<u8>),
    Array(Vec<Value>),
    Struct(BTreeMap<String, Value>),
}

impl Value {
    pub fn field(&self, name: &str) -> Option<&Self> {
        match self {
            Self::Struct(fields) => fields.get(name),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

impl HailType {
    pub fn parse(text: &str) -> Result<Self> {
        let mut parser = Parser {
            input: text.as_bytes(),
            offset: 0,
        };
        let result = parser.typ(0)?;
        parser.space();
        if parser.offset != parser.input.len() {
            return Err(invalid(format!("trailing type text at {}", parser.offset)));
        }
        Ok(result)
    }

    fn simple(name: &str) -> Self {
        Self {
            name: name.into(),
            required: true,
            fields: vec![],
            args: vec![],
        }
    }

    fn logical_field(&self, name: &str, index: usize) -> Option<Self> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.clone())
            .or_else(|| match self.name.as_str() {
                "Locus" => Some(Self::simple(if name == "contig" {
                    "String"
                } else {
                    "Int32"
                })),
                "Interval" if index < 2 => self.args.first().cloned(),
                "Interval" => Some(Self::simple("Boolean")),
                "Tuple" => self.args.get(index).cloned(),
                _ => None,
            })
    }

    pub(crate) fn decode(&self, cursor: &mut Cursor<'_>, logical: Option<&Self>) -> Result<Value> {
        Ok(match self.name.as_str() {
            "EBoolean" => match cursor.byte()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                n => return Err(invalid(format!("invalid boolean {n}"))),
            },
            "EInt32" => Value::Int(cursor.int()? as i64),
            "EInt64" => Value::Int(cursor.long()?),
            "EVarint" => {
                let bits = match logical.map(|t| t.name.as_str()) {
                    Some("Int64") => 64,
                    Some("Locus") => {
                        let reference = &logical
                            .unwrap()
                            .args
                            .first()
                            .ok_or_else(|| invalid("packed locus reference"))?
                            .name;
                        return crate::reference::unpack_locus(reference, cursor.varint(64)?);
                    }
                    Some("Int32" | "Call") => 32,
                    _ => {
                        return Err(Error::Unsupported(
                            "EVarint needs an integral logical type".into(),
                        ))
                    }
                };
                let value = cursor.varint(bits)?;
                Value::Int(if bits == 32 {
                    value as i32 as i64
                } else {
                    value as i64
                })
            }
            "EFloat32" => Value::Float(f32::from_bits(cursor.u32()?) as f64),
            "EFloat64" => Value::Float(f64::from_bits(cursor.u64()?)),
            "EBinary" | "EBinary2" | "EBinaryLegacyFullWidthIntegerLength" => {
                let n = cursor.length(self.name == "EBinary2")?;
                let bytes = cursor.take(n)?;
                if logical.is_some_and(|t| t.name == "String") {
                    Value::String(
                        std::str::from_utf8(bytes)
                            .map_err(|e| invalid(format!("UTF-8: {e}")))?
                            .into(),
                    )
                } else {
                    Value::Binary(bytes.to_vec())
                }
            }
            "EBaseStruct" => {
                let optional = self.fields.iter().filter(|(_, t)| !t.required).count();
                let missing = cursor.take(optional.div_ceil(8))?.to_vec();
                let mut bit = 0;
                let mut fields = BTreeMap::new();
                for (index, (name, typ)) in self.fields.iter().enumerate() {
                    let is_missing = !typ.required && missing[bit / 8] & (1 << (bit % 8)) != 0;
                    if !typ.required {
                        bit += 1;
                    }
                    let value = if is_missing {
                        Value::Null
                    } else {
                        typ.decode(
                            cursor,
                            logical.and_then(|t| t.logical_field(name, index)).as_ref(),
                        )?
                    };
                    fields.insert(name.clone(), value);
                }
                Value::Struct(fields)
            }
            "EArray" | "EArray2" | "EArrayLegacyFullWidthIntegerLength" => {
                let n = cursor.length(self.name == "EArray2")?;
                let element = self
                    .args
                    .first()
                    .ok_or_else(|| invalid("array element type"))?;
                let missing = cursor
                    .take(if element.required { 0 } else { n.div_ceil(8) })?
                    .to_vec();
                let element_logical = logical.and_then(|t| {
                    if t.name == "Dict" && t.args.len() == 2 {
                        Some(Self {
                            name: "Struct".into(),
                            required: true,
                            fields: vec![
                                ("key".into(), t.args[0].clone()),
                                ("value".into(), t.args[1].clone()),
                            ],
                            args: vec![],
                        })
                    } else {
                        t.args.first().cloned()
                    }
                });
                let mut values = Vec::new();
                for i in 0..n {
                    values.push(
                        if !element.required && missing[i / 8] & (1 << (i % 8)) != 0 {
                            Value::Null
                        } else {
                            element.decode(cursor, element_logical.as_ref())?
                        },
                    );
                }
                Value::Array(values)
            }
            "ENDArrayColumnMajor" => {
                let dims = self
                    .args
                    .get(1)
                    .ok_or_else(|| invalid("ndarray rank"))?
                    .name
                    .parse::<usize>()
                    .map_err(|_| invalid("ndarray rank"))?;
                if dims > 32 {
                    return Err(invalid("ndarray rank > 32"));
                }
                let mut shape = Vec::new();
                let mut count = 1usize;
                for _ in 0..dims {
                    let n = cursor.long()?;
                    if n < 0 {
                        return Err(invalid("negative ndarray dimension"));
                    }
                    count = count
                        .checked_mul(n as usize)
                        .filter(|n| *n <= 16_777_216)
                        .ok_or_else(|| invalid("ndarray too large"))?;
                    shape.push(Value::Int(n));
                }
                let mut data = Vec::new();
                for _ in 0..count {
                    data.push(self.args[0].decode(cursor, logical.and_then(|t| t.args.first()))?);
                }
                Value::Struct(
                    [
                        ("shape".into(), Value::Array(shape)),
                        ("data".into(), Value::Array(data)),
                    ]
                    .into(),
                )
            }
            other => return Err(Error::Unsupported(format!("encoded type {other}"))),
        })
    }
}

struct Parser<'a> {
    input: &'a [u8],
    offset: usize,
}

impl Parser<'_> {
    fn space(&mut self) {
        while self
            .input
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }
    fn eat(&mut self, byte: u8) -> bool {
        self.space();
        if self.input.get(self.offset) == Some(&byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, byte: u8) -> Result<()> {
        if self.eat(byte) {
            Ok(())
        } else {
            Err(invalid(format!(
                "expected '{}' at type byte {}",
                byte as char, self.offset
            )))
        }
    }
    fn ident(&mut self) -> Result<String> {
        self.space();
        if self.eat(b'`') {
            let mut out = Vec::new();
            loop {
                let c = *self
                    .input
                    .get(self.offset)
                    .ok_or_else(|| invalid("unclosed quoted identifier"))?;
                self.offset += 1;
                if c == b'`' {
                    break;
                }
                if c == b'\\' {
                    let escaped = *self
                        .input
                        .get(self.offset)
                        .ok_or_else(|| invalid("identifier escape"))?;
                    self.offset += 1;
                    out.push(escaped);
                } else {
                    out.push(c);
                }
            }
            return String::from_utf8(out).map_err(|e| invalid(e.to_string()));
        }
        let start = self.offset;
        while self
            .input
            .get(self.offset)
            .is_some_and(|c| c.is_ascii_alphanumeric() || b"_.-".contains(c))
        {
            self.offset += 1;
        }
        if start == self.offset {
            return Err(invalid(format!("expected identifier at {}", self.offset)));
        }
        String::from_utf8(self.input[start..self.offset].to_vec())
            .map_err(|e| invalid(e.to_string()))
    }
    fn typ(&mut self, depth: usize) -> Result<HailType> {
        if depth > 128 {
            return Err(invalid("type nesting exceeds 128"));
        }
        let required = self.eat(b'+');
        let name = self.ident()?;
        let mut result = HailType {
            name,
            required,
            fields: vec![],
            args: vec![],
        };
        if self.eat(b'{') {
            if !self.eat(b'}') {
                loop {
                    let name = self.ident()?;
                    self.expect(b':')?;
                    let typ = self.typ(depth + 1)?;
                    if result.fields.iter().any(|(n, _)| n == &name) {
                        return Err(invalid("duplicate field"));
                    }
                    result.fields.push((name, typ));
                    if self.eat(b'}') {
                        break;
                    }
                    self.expect(b',')?;
                }
            }
        } else {
            let close = if self.eat(b'[') {
                Some(b']')
            } else if self.eat(b'(') {
                Some(b')')
            } else {
                None
            };
            if let Some(close) = close {
                if !self.eat(close) {
                    loop {
                        result.args.push(self.typ(depth + 1)?);
                        if self.eat(close) {
                            break;
                        }
                        self.expect(b',')?;
                    }
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_and_quoted_types() {
        let t = HailType::parse(
            "+EBaseStruct{`a b`:EArray[+EInt32],l:EBaseStruct{contig:+EBinary,position:+EInt32}}",
        )
        .unwrap();
        assert!(t.required);
        assert_eq!(t.fields[0].0, "a b");
        assert!(HailType::parse("EInt32 junk").is_err());
        assert!(HailType::parse("EBaseStruct{x:EInt32,x:EInt64}").is_err());
    }
    #[test]
    fn missing_bits_and_negative_varints() {
        let t = HailType::parse("+EBaseStruct{a:EInt32,b:+EInt64,c:EArray[EInt32]}").unwrap();
        let data = [
            1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 2, 1, 42,
        ];
        let value = t.decode(&mut Cursor::new(&data, true), None).unwrap();
        assert_eq!(value.field("a"), Some(&Value::Null));
        assert_eq!(value.field("b"), Some(&Value::Int(-1)));
        assert_eq!(
            value.field("c"),
            Some(&Value::Array(vec![Value::Null, Value::Int(42)]))
        );
    }

    #[test]
    fn modern_encoding_without_leb_wrapper() {
        let encoded =
            HailType::parse("+EBaseStruct{n:+EVarint,s:+EBinary2,a:+EArray2[+EVarint]}").unwrap();
        let logical = HailType::parse("Struct{n:Int64,s:String,a:Array[Int32]}").unwrap();
        let data = [172, 2, 2, b'h', b'i', 2, 1, 2];
        let row = encoded
            .decode(&mut Cursor::new(&data, false), Some(&logical))
            .unwrap();
        assert_eq!(row.field("n"), Some(&Value::Int(300)));
        assert_eq!(row.field("s"), Some(&Value::String("hi".into())));
        assert_eq!(
            row.field("a"),
            Some(&Value::Array(vec![Value::Int(1), Value::Int(2)]))
        );
    }

    #[test]
    fn ndarray_and_binary() {
        let encoded = HailType::parse("ENDArrayColumnMajor[+EFloat64,2]").unwrap();
        let mut data = Vec::new();
        data.extend(2i64.to_le_bytes());
        data.extend(1i64.to_le_bytes());
        data.extend(1.0f64.to_le_bytes());
        data.extend(f64::NAN.to_le_bytes());
        let row = encoded
            .decode(&mut Cursor::new(&data, false), None)
            .unwrap();
        assert_eq!(
            row.field("shape"),
            Some(&Value::Array(vec![Value::Int(2), Value::Int(1)]))
        );
        assert_eq!(
            HailType::parse("EBinary2")
                .unwrap()
                .decode(&mut Cursor::new(&[2, 0, 255], false), None)
                .unwrap(),
            Value::Binary(vec![0, 255])
        );
    }
}
