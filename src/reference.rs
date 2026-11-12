use crate::{invalid, Error, Result, Value};
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct Reference {
    name: String,
    contigs: Vec<Contig>,
}

#[derive(Deserialize)]
struct Contig {
    name: String,
    length: i64,
}

fn reference(name: &str) -> Option<&'static Reference> {
    static REFERENCES: OnceLock<[Reference; 2]> = OnceLock::new();
    REFERENCES
        .get_or_init(|| {
            [
                serde_json::from_str(include_str!("reference/grch37.json"))
                    .expect("embedded GRCh37 reference"),
                serde_json::from_str(include_str!("reference/grch38.json"))
                    .expect("embedded GRCh38 reference"),
            ]
        })
        .iter()
        .find(|reference| reference.name == name)
}

pub(crate) fn contig_rank(reference_name: &str, contig: &str) -> Option<usize> {
    reference(reference_name)?
        .contigs
        .iter()
        .position(|c| c.name == contig)
}

pub(crate) fn unpack_locus(reference_name: &str, packed: u64) -> Result<Value> {
    let reference = reference(reference_name).ok_or_else(|| {
        Error::Unsupported(format!("packed loci on custom reference {reference_name}"))
    })?;
    let contig = reference
        .contigs
        .get((packed >> 32) as usize)
        .ok_or_else(|| invalid("packed locus contig index"))?;
    let position = (packed as u32) as i64;
    if position < 1 || position > contig.length {
        return Err(invalid("packed locus position outside contig"));
    }

    Ok(Value::Struct(
        [
            ("contig".into(), Value::String(contig.name.clone())),
            ("position".into(), Value::Int(position)),
        ]
        .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_loci_and_alternate_contig_order() {
        let packed = (21u64 << 32) | 20012563;
        let value = unpack_locus("GRCh38", packed).unwrap();
        assert_eq!(value.field("contig"), Some(&Value::String("chr22".into())));
        assert_eq!(value.field("position"), Some(&Value::Int(20012563)));
        assert!(contig_rank("GRCh38", "chr4_GL000008v2_random") > contig_rank("GRCh38", "chr22"));
        assert!(unpack_locus("GRCh38", u64::MAX).is_err());
        assert!(unpack_locus("custom", packed).is_err());
    }
}
