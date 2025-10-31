use clap::Parser;
use num_bigint::{BigUint, RandBigInt};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::rngs::OsRng;
use rand::Rng;
use std::convert::TryFrom;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::thread;

/// Default Miller–Rabin rounds for probabilistic primality checks.
const DEFAULT_MR_ROUNDS: usize = 40;

/// Command-line arguments for the safe prime generator.
#[derive(Parser, Debug)]
#[command(
    name = "find-big-safe-prime",
    about = "Generate large safe primes p = 2q + 1 and print both p and q"
)]
struct Args {
    /// Bit width for the safe prime. Defaults to 2048.
    #[arg(short = 'b', long = "bits", default_value_t = 2048)]
    bits: usize,

    /// Number of Miller–Rabin rounds to execute.
    #[arg(long = "rounds", default_value_t = DEFAULT_MR_ROUNDS)]
    rounds: usize,

    /// Number of worker threads to search with (defaults to available CPUs).
    #[arg(long = "threads")]
    threads: Option<usize>,
}

fn main() {
    let args = Args::parse();
    assert!(
        args.bits >= 512,
        "At least 512 bits are recommended; production use typically >= 2048 bits."
    );

    let default_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let threads = args.threads.unwrap_or(default_threads).max(1);

    let (p, q) = if threads <= 1 {
        generate_safe_prime(args.bits, args.rounds)
    } else {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(threads);

        for _ in 0..threads {
            let tx_clone = tx.clone();
            let stop_clone = Arc::clone(&stop);
            let bits = args.bits;
            let rounds = args.rounds;
            handles.push(thread::spawn(move || {
                let mut rng = OsRng;
                while !stop_clone.load(Ordering::Relaxed) {
                    let result = generate_safe_prime_with_rng(bits, rounds, &mut rng);
                    if stop_clone.load(Ordering::Relaxed) {
                        break;
                    }
                    if tx_clone.send(result).is_ok() {
                        stop_clone.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }));
        }

        drop(tx);
        let result = rx.recv().expect("failed to receive safe prime");
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            let _ = handle.join();
        }
        result
    };

    println!("safe_prime_bits={}", p.bits());
    println!("p={p}");
    println!("q_bits={}", q.bits());
    println!("q={q}");
}

/// Generate a safe prime p = 2q + 1 where p and q are probable primes.
fn generate_safe_prime(bits: usize, rounds: usize) -> (BigUint, BigUint) {
    let mut rng = OsRng;
    generate_safe_prime_with_rng(bits, rounds, &mut rng)
}

fn generate_safe_prime_with_rng<R>(bits: usize, rounds: usize, rng: &mut R) -> (BigUint, BigUint)
where
    R: Rng + ?Sized,
{
    assert!(bits >= 3, "Safe primes require at least 3 bits.");
    let q_bits = bits - 1;
    loop {
        let q = generate_probable_prime_with_rng(q_bits, rounds, rng);
        let p = (&q << 1usize) + BigUint::one();
        if is_probable_prime_with_rng(&p, rounds, rng) {
            return (p, q);
        }
    }
}

fn generate_probable_prime_with_rng<R>(bits: usize, rounds: usize, rng: &mut R) -> BigUint
where
    R: Rng + ?Sized,
{
    let bits_u64 = u64::try_from(bits).expect("bit size must fit in u64");
    loop {
        let mut n = rng.gen_biguint(bits_u64);
        let one = BigUint::one();

        // Force highest bit to ensure bit length and make the candidate odd.
        n.set_bit(bits_u64 - 1, true);
        if n.is_even() {
            n |= &one;
        }

        if !small_prime_precheck(&n) {
            continue;
        }

        if is_probable_prime_with_rng(&n, rounds, rng) {
            return n;
        }
    }
}

fn is_probable_prime_with_rng<R>(n: &BigUint, rounds: usize, rng: &mut R) -> bool
where
    R: Rng + ?Sized,
{
    let two = BigUint::from(2u32);

    if *n < two {
        return false;
    }
    if *n == two {
        return true;
    }
    if n.is_even() {
        return false;
    }

    let one = BigUint::one();
    let n_minus_one = n - &one;
    let (s, d) = factor_out_twos(&n_minus_one);

    'witness: for _ in 0..rounds {
        let a = random_range(&two, &n_minus_one, rng);
        let mut x = a.modpow(&d, n);

        if x == one || x == n_minus_one {
            continue 'witness;
        }

        for _ in 1..s {
            x = x.modpow(&two, n);
            if x == n_minus_one {
                continue 'witness;
            }
            if x == one {
                return false;
            }
        }

        return false;
    }

    true
}

/// Express n as d * 2^s with d odd, returning (s, d).
fn factor_out_twos(n: &BigUint) -> (u32, BigUint) {
    let mut s = 0u32;
    let mut d = n.clone();
    while d.is_even() {
        d >>= 1;
        s += 1;
    }
    (s, d)
}

/// Sample a random value in the inclusive range [low, high].
fn random_range<R>(low: &BigUint, high: &BigUint, rng: &mut R) -> BigUint
where
    R: Rng + ?Sized,
{
    if low == high {
        return low.clone();
    }
    let range = high - low + BigUint::one();
    let k = rng.gen_biguint_below(&range);
    low + k
}

/// Filter out obvious composites using a small set of primes.
fn small_prime_precheck(n: &BigUint) -> bool {
    const SMALLS: [u32; 16] = [
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
    ];

    if n == &BigUint::one() {
        return false;
    }

    for &p in SMALLS.iter() {
        let p_big = BigUint::from(p);
        if n == &p_big {
            return true;
        }
        if (n % &p_big).is_zero() {
            return false;
        }
    }

    true
}
