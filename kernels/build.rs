use std::env;
use std::fs::{File, read_to_string};
use std::io::Write;
use std::path::PathBuf;

fn main() {
    // Generate primes array at compile time
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let primes_file = out_dir.join("primes.rs");

    // Generate 140,741 primes (same as cuCollections)
    // The largest prime in cuCollections is around 17 billion
    const NUM_PRIMES: usize = 140_741;
    const MAX_PRIME: usize = 17_177_581_333; // Approximate max from cuCollections

    println!("cargo:rerun-if-changed=build.rs");

    // Check if file already exists and is valid (contains expected number of primes)
    if primes_file.exists() {
        if let Ok(content) = read_to_string(&primes_file) {
            // Check if it contains the expected number of primes
            let prime_count = content.matches(',').count() + 1; // Count commas + 1
            if prime_count == NUM_PRIMES && content.contains("pub const PRIMES") {
                println!(
                    "cargo:warning=Using existing primes.rs with {} primes",
                    NUM_PRIMES
                );
                return; // Skip regeneration
            }
        }
    }

    let mut file = File::create(&primes_file).expect("Failed to create primes.rs");

    writeln!(file, "// Auto-generated primes array for double hashing").unwrap();
    writeln!(file, "// Generated at compile time using primal crate").unwrap();
    writeln!(
        file,
        "// Contains {} primes up to approximately {}",
        NUM_PRIMES, MAX_PRIME
    )
    .unwrap();
    writeln!(file, "").unwrap();
    writeln!(file, "pub const PRIMES: &[usize] = &[").unwrap();

    // Use primal to generate primes
    let sieve = primal::Sieve::new(MAX_PRIME);
    let primes: Vec<usize> = sieve.primes_from(2).take(NUM_PRIMES).collect();

    // Write primes in chunks for readability
    const PRIMES_PER_LINE: usize = 10;
    for (i, &prime) in primes.iter().enumerate() {
        if i % PRIMES_PER_LINE == 0 {
            if i > 0 {
                writeln!(file, "").unwrap();
            }
            write!(file, "    ").unwrap();
        }
        write!(file, "{}", prime).unwrap();
        if i < primes.len() - 1 {
            write!(file, ", ").unwrap();
        }
    }

    writeln!(file, "").unwrap();
    writeln!(file, "];").unwrap();

    println!("cargo:warning=Generated {} primes", primes.len());
}
