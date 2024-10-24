#![feature(allocator_api)]
use std::time::{Duration, Instant};

use utils::benchmark;

use std::fs::File;
use std::io::BufReader;
use std::ops::Range;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;

use zkm_emulator::utils::{load_elf_with_patch, split_prog_into_segs};
use zkm_prover::all_stark::AllStark;
use zkm_prover::config::StarkConfig;
use zkm_prover::cpu::kernel::assembler::segment_kernel;
use zkm_prover::fixed_recursive_verifier::AllRecursiveCircuits;
use zkm_prover::proof;
use zkm_prover::proof::PublicValues;
#[cfg(not(feature = "gpu"))]
use zkm_prover::prover::prove;
use zkm_prover::verifier::verify_proof;

#[cfg(feature = "gpu")]
use zkm_prover::prover::prove_gpu;
#[cfg(feature = "gpu")]
use rustacuda::{
    memory::DeviceBuffer, prelude::*,
};
#[cfg(feature = "gpu")]
use plonky2::{
    plonk::config::Hasher,
    field::fft::fft_root_table,
    field::types::Field,
    field::extension::Extendable,
    fri::oracle::{CudaInnerContext, MyAllocator, create_task},
};
#[cfg(feature = "gpu")]
use std::{
    collections::BTreeMap, sync::Arc,
};

const FIBONACCI_ELF: &str = "./fibonacci/target/mips-unknown-linux-musl/release/fibonacci";
const SHA2_ELF: &str = "./sha2/target/mips-unknown-linux-musl/release/sha2-bench";
const SHA2_CHAIN_ELF: &str = "./sha2-chain/target/mips-unknown-linux-musl/release/sha2-chain";
const SHA3_CHAIN_ELF: &str = "./sha3-chain/target/mips-unknown-linux-musl/release/sha3-chain";
const SHA3_ELF: &str = "./sha3/target/mips-unknown-linux-musl/release/sha3-bench";
const BIGMEM_ELF: &str = "./bigmem/target/mips-unknown-linux-musl/release/bigmem";
const SEG_SIZE: usize = 262144 * 8; //G

const DEGREE_BITS_RANGE: [Range<usize>; 6] = [10..23, 10..23, 10..23, 8..23, 6..23, 13..25];

fn main() {
    init_logger();

    let _ = std::fs::remove_dir_all("/tmp/zkm.old");
    let _ = std::fs::rename("/tmp/zkm", "/tmp/zkm.old");

    let lengths = [32, 256, 512, 1024, 2048];
    benchmark(benchmark_sha2, &lengths, "../benchmark_outputs/sha2_zkm.csv", "byte length");
    benchmark(benchmark_sha3, &lengths, "../benchmark_outputs/sha3_zkm.csv", "byte length");

    let ns = [100, 1000, 10000, 50000];
    benchmark(benchmark_fibonacci, &ns, "../benchmark_outputs/fiboancci_zkm.csv", "n");

    let values = [5];
    benchmark(benchmark_bigmem, &values, "../benchmark_outputs/bigmem_zkm.csv", "value");

    let iters = [230, 460, 920, 1840, 3680];
    benchmark(benchmark_sha2_chain, &iters, "../benchmark_outputs/sha2_chain_zkm.csv", "iters");
    benchmark(benchmark_sha3_chain, &iters, "../benchmark_outputs/sha3_chain_zkm.csv", "iters");
}

fn prove_single_seg_common(seg_file: &str, basedir: &str, block: &str, file: &str) -> usize {
    #[cfg(feature = "gpu")]
    prove_single_seg_gpu(seg_file, basedir, block, file)

    #[cfg(not(feature = "gpu"))]
    prove_single_seg_cpu(seg_file, basedir, block, file)
}

#[cfg(not(feature = "gpu"))]
fn prove_single_seg_cpu(
    seg_file: &str,
    basedir: &str,
    block: &str,
    file: &str,
) -> usize {
    let seg_reader = BufReader::new(File::open(seg_file).unwrap());
    let kernel = segment_kernel(basedir, block, file, seg_reader);

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let allstark: AllStark<F, D> = AllStark::default();
    let config = StarkConfig::standard_fast_config();
    let mut timing = TimingTree::new("prove", log::Level::Info);
    let allproof: proof::AllProof<GoldilocksField, C, D> =
        prove(&allstark, &kernel, &config, &mut timing).unwrap();
    let mut count_bytes: usize = 0;
    for (row, proof) in allproof.stark_proofs.clone().iter().enumerate() {
        let proof_str = serde_json::to_string(&proof.proof).unwrap();
        log::info!("row:{} proof bytes:{}", row, proof_str.len());
        count_bytes += proof_str.len();
    }
    timing.filter(Duration::from_millis(100)).print();
    log::info!("total proof bytes:{}KB", count_bytes / 1024);
    verify_proof(&allstark, allproof, &config).unwrap();
    log::info!("Prove done");
    count_bytes
}

#[cfg(feature = "gpu")]
fn prove_single_seg_gpu(seg_file: &str, basedir: &str, block: &str, file: &str) -> usize {
    let seg_reader = BufReader::new(File::open(seg_file).unwrap());
    let kernel = segment_kernel(&basedir, &block, &file, seg_reader);

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let allstark: AllStark<F, D> = AllStark::default();
    let config = StarkConfig::standard_fast_config();

    let mut ctx: plonky2::fri::oracle::CudaInvContext<GoldilocksField, C, D>;
    {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device_index = 0;
        let device = Device::get_device(device_index).unwrap();
        let _ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let stream2 = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        let rate_bits = config.fri_config.rate_bits;
        let blinding = false;
        const SALT_SIZE: usize = 4;
        let cap_height = config.fri_config.cap_height;
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        // let max_lg_n = tasks.iter().max_by_key(|kv|kv.1.lg_n).unwrap().1.lg_n;
        // println!("max_lg_n: {}", max_lg_n);
        let max_lg_n = 22;
        let fft_root_table_max = fft_root_table(1 << (max_lg_n + rate_bits)).concat();
        let root_table_device = {
            DeviceBuffer::from_slice(&fft_root_table_max).unwrap() };

        let fft_root_table_ext = fft_root_table::<<F as Extendable<{ D }>>::Extension>(1 << (24)).concat();
        let root_table_ext_device = {
            DeviceBuffer::from_slice(&fft_root_table_ext).unwrap() };

        let shift_powers = F::coset_shift()
            .powers()
            .take(1 << (max_lg_n))
            .collect::<Vec<_>>();
        let shift_powers_device = {
            DeviceBuffer::from_slice(&shift_powers).unwrap() };

        let shift_powers_ext = <<F as Extendable<{ D }>>::Extension>::coset_shift()
            .powers()
            .take(1 << (22))
            .collect::<Vec<_>>();
        let shift_powers_ext_device = {
            DeviceBuffer::from_slice(&shift_powers_ext).unwrap() };

        let max_values_num_per_poly = 1 << max_lg_n;
        let max_values_flatten_len = max_values_num_per_poly * 32;
        // let max_values_flatten_len = 132644864;
        let max_ext_values_flatten_len =
            (max_values_flatten_len + salt_size * max_values_num_per_poly) * (1 << rate_bits);
        let mut ext_values_flatten: Vec<F> = Vec::with_capacity(max_ext_values_flatten_len);
        unsafe {
            ext_values_flatten.set_len(max_ext_values_flatten_len);
        }

        let mut values_flatten: Vec<F, MyAllocator> =
            Vec::with_capacity_in(max_values_flatten_len, MyAllocator {});
        unsafe {
            values_flatten.set_len(max_values_flatten_len);
        }

        let len_cap = 1 << cap_height;
        let num_digests = 2 * (max_values_num_per_poly * (1 << rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;
        let mut digests_and_caps_buf: Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash> =
            Vec::with_capacity(num_digests_and_caps);
        unsafe {
            digests_and_caps_buf.set_len(num_digests_and_caps);
        }

        let pad_extvalues_len = max_ext_values_flatten_len;
        let cache_mem_device = {
            unsafe {
                DeviceBuffer::<F>::uninitialized(
                    // values_flatten_len +
                    pad_extvalues_len + max_ext_values_flatten_len + digests_and_caps_buf.len() * 4,
                )
            }
            .unwrap()
        };

        ctx = plonky2::fri::oracle::CudaInvContext {
            inner: CudaInnerContext { stream, stream2 },
            ext_values_flatten: Arc::new(ext_values_flatten),
            values_flatten: Arc::new(values_flatten),
            digests_and_caps_buf: Arc::new(digests_and_caps_buf),
            cache_mem_device,
            root_table_device,
            shift_powers_device,
            // cache_mem_ext_device,
            root_table_ext_device,
            shift_powers_ext_device,
            tasks: BTreeMap::new(),
            ctx: _ctx,
        };
    }

    create_task(&mut ctx, 0, 17, 54, 0, 2, 4);
    create_task(&mut ctx, 1, 18, 256, 0, 2, 4);
    create_task(&mut ctx, 2, 15, 262, 0, 2, 4);
    create_task(&mut ctx, 3, 15, 110, 0, 2, 4);
    create_task(&mut ctx, 4, 15, 69, 0, 2, 4);
    create_task(&mut ctx, 5, 21, 13, 0, 2, 4);
    create_task(&mut ctx, 6, 17, 22, 0, 2, 4);
    create_task(&mut ctx, 12, 17, 4, 0, 2, 4);
    create_task(&mut ctx, 7, 18, 20, 0, 2, 4);
    create_task(&mut ctx, 13, 18, 4, 0, 2, 4);
    create_task(&mut ctx, 14, 15, 4, 0, 2, 4);
    create_task(&mut ctx, 9, 15, 40, 0, 2, 4);
    create_task(&mut ctx, 15, 15, 4, 0, 2, 4);
    create_task(&mut ctx, 16, 15, 4, 0, 2, 4);
    create_task(&mut ctx, 11, 21, 6, 0, 2, 4);
    create_task(&mut ctx, 17, 21, 4, 0, 2, 4);

    let mut timing = TimingTree::new("prove", log::Level::Info);
    let allproof: proof::AllProof<GoldilocksField, C, D> =
        prove_gpu(&allstark, &kernel, &config, &mut timing, &mut ctx).unwrap();


    let mut count_bytes = 0;
    for (row, proof) in allproof.stark_proofs.clone().iter().enumerate() {
        let proof_str = serde_json::to_string(&proof.proof).unwrap();
        log::info!("row:{} proof bytes:{}", row, proof_str.len());
        count_bytes += proof_str.len();
    }
    // timing.filter(Duration::from_millis(100)).print();
    timing.print();

    log::info!("total proof bytes:{}KB", count_bytes / 1024);
    verify_proof(&allstark, allproof, &config).unwrap();
    log::info!("Prove done");
    count_bytes
}

fn prove_multi_seg_common(
    seg_dir: &str,
    basedir: &str,
    block: &str,
    file: &str,
    seg_file_number: usize,
    seg_start_id: usize,
) -> usize {
    type F = GoldilocksField;
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;

    if seg_file_number < 2 {
        panic!("seg file number must >= 2\n");
    }

    let total_timing = TimingTree::new("prove total time", log::Level::Info);
    let all_stark = AllStark::<F, D>::default();
    let config = StarkConfig::standard_fast_config();
    // Preprocess all circuits.
    let all_circuits =
        AllRecursiveCircuits::<F, C, D>::new(&all_stark, &DEGREE_BITS_RANGE, &config);

    let seg_file = format!("{}/{}", seg_dir, seg_start_id);
    log::info!("Process segment {}", seg_file);
    let seg_reader = BufReader::new(File::open(seg_file).unwrap());
    let input_first = segment_kernel(basedir, block, file, seg_reader);
    let mut timing = TimingTree::new("prove root first", log::Level::Info);
    let (mut agg_proof, mut updated_agg_public_values) =
        all_circuits.prove_root(&all_stark, &input_first, &config, &mut timing).unwrap();

    timing.filter(Duration::from_millis(100)).print();
    all_circuits.verify_root(agg_proof.clone()).unwrap();

    let mut base_seg = seg_start_id + 1;
    let mut seg_num = seg_file_number - 1;
    let mut is_agg = false;

    if seg_file_number % 2 == 0 {
        let seg_file = format!("{}/{}", seg_dir, seg_start_id + 1);
        log::info!("Process segment {}", seg_file);
        let seg_reader = BufReader::new(File::open(seg_file).unwrap());
        let input = segment_kernel(basedir, block, file, seg_reader);
        timing = TimingTree::new("prove root second", log::Level::Info);
        let (root_proof, public_values) =
            all_circuits.prove_root(&all_stark, &input, &config, &mut timing).unwrap();
        timing.filter(Duration::from_millis(100)).print();

        all_circuits.verify_root(root_proof.clone()).unwrap();

        // Update public values for the aggregation.
        let agg_public_values = PublicValues {
            roots_before: updated_agg_public_values.roots_before,
            roots_after: public_values.roots_after,
            userdata: public_values.userdata,
        };
        timing = TimingTree::new("prove aggression", log::Level::Info);
        // We can duplicate the proofs here because the state hasn't mutated.
        (agg_proof, updated_agg_public_values) = all_circuits.prove_aggregation(
            false,
            &agg_proof,
            false,
            &root_proof,
            agg_public_values.clone(),
        ).unwrap();
        timing.filter(Duration::from_millis(100)).print();
        all_circuits.verify_aggregation(&agg_proof).unwrap();

        is_agg = true;
        base_seg = seg_start_id + 2;
        seg_num -= 1;
    }

    for i in 0..seg_num / 2 {
        let seg_file = format!("{}/{}", seg_dir, base_seg + (i << 1));
        log::info!("Process segment {}", seg_file);
        let seg_reader = BufReader::new(File::open(&seg_file).unwrap());
        let input_first = segment_kernel(basedir, block, file, seg_reader);
        let mut timing = TimingTree::new("prove root first", log::Level::Info);
        let (root_proof_first, first_public_values) =
            all_circuits.prove_root(&all_stark, &input_first, &config, &mut timing).unwrap();

        timing.filter(Duration::from_millis(100)).print();
        all_circuits.verify_root(root_proof_first.clone()).unwrap();

        let seg_file = format!("{}/{}", seg_dir, base_seg + (i << 1) + 1);
        log::info!("Process segment {}", seg_file);
        let seg_reader = BufReader::new(File::open(&seg_file).unwrap());
        let input = segment_kernel(basedir, block, file, seg_reader);
        let mut timing = TimingTree::new("prove root second", log::Level::Info);
        let (root_proof, public_values) =
            all_circuits.prove_root(&all_stark, &input, &config, &mut timing).unwrap();
        timing.filter(Duration::from_millis(100)).print();

        all_circuits.verify_root(root_proof.clone()).unwrap();

        // Update public values for the aggregation.
        let new_agg_public_values = PublicValues {
            roots_before: first_public_values.roots_before,
            roots_after: public_values.roots_after,
            userdata: public_values.userdata,
        };
        timing = TimingTree::new("prove aggression", log::Level::Info);
        // We can duplicate the proofs here because the state hasn't mutated.
        let (new_agg_proof, new_updated_agg_public_values) = all_circuits.prove_aggregation(
            false,
            &root_proof_first,
            false,
            &root_proof,
            new_agg_public_values,
        ).unwrap();
        timing.filter(Duration::from_millis(100)).print();
        all_circuits.verify_aggregation(&new_agg_proof).unwrap();

        // Update public values for the nested aggregation.
        let agg_public_values = PublicValues {
            roots_before: updated_agg_public_values.roots_before,
            roots_after: new_updated_agg_public_values.roots_after,
            userdata: new_updated_agg_public_values.userdata,
        };
        timing = TimingTree::new("prove nested aggression", log::Level::Info);

        // We can duplicate the proofs here because the state hasn't mutated.
        (agg_proof, updated_agg_public_values) = all_circuits.prove_aggregation(
            is_agg,
            &agg_proof,
            true,
            &new_agg_proof,
            agg_public_values.clone(),
        ).unwrap();
        is_agg = true;
        timing.filter(Duration::from_millis(100)).print();

        all_circuits.verify_aggregation(&agg_proof).unwrap();
    }

    let (block_proof, _block_public_values) =
        all_circuits.prove_block(None, &agg_proof, updated_agg_public_values).unwrap();

    let size = serde_json::to_string(&block_proof.proof).unwrap().len();
    log::info!(
        "proof size: {:?}",
        size
    );
    total_timing.filter(Duration::from_millis(100)).print();
    size
}

fn init_logger() {
    let logl = std::env::var("RUST_LOG").unwrap_or("info".to_string());
    std::env::set_var("RUST_LOG", &logl);
    env_logger::init()
}

fn benchmark_sha2_chain(iters: u32) -> (Duration, usize) {
    let input = [5u8; 32];
    let mut state = load_elf_with_patch(SHA2_CHAIN_ELF, vec![]);
    state.add_input_stream(&input);
    state.add_input_stream(&iters);

    let seg_size = SEG_SIZE;
    let seg_path = "/tmp/zkm/sha2-chain";

    let (_total_steps, seg_num, mut state) = split_prog_into_segs(state, seg_path, "", seg_size);

    let start = Instant::now();
    let size = if seg_num == 1 {
        let seg_file = format!("{seg_path}/{}", 0);
        prove_single_seg_common(&seg_file, "", "", "")
    } else {
        prove_multi_seg_common(seg_path, "", "", "", seg_num, 0)
    };
    let end = Instant::now();
    let duration = end.duration_since(start);

    let _hash =  state.read_public_values::<[u8; 32]>();

    (duration, size)
}

fn benchmark_sha3_chain(iters: u32) -> (Duration, usize) {
    let input = [5u8; 32];
    let mut state = load_elf_with_patch(SHA3_CHAIN_ELF, vec![]);
    state.add_input_stream(&input);
    state.add_input_stream(&iters);

    let seg_size = SEG_SIZE;
    let seg_path = "/tmp/zkm/sha3-chain";

    let (_total_steps, seg_num, mut state) = split_prog_into_segs(state, seg_path, "", seg_size);

    let start = Instant::now();
    let size = if seg_num == 1 {
        let seg_file = format!("{seg_path}/{}", 0);
        prove_single_seg_common(&seg_file, "", "", "")
    } else {
        prove_multi_seg_common(seg_path, "", "", "", seg_num, 0)
    };
    let end = Instant::now();
    let duration = end.duration_since(start);

    let _hash =  state.read_public_values::<[u8; 32]>();

    (duration, size)
}

fn benchmark_sha2(num_bytes: usize) -> (Duration, usize) {
    let input = vec![5u8; num_bytes];
    let mut state = load_elf_with_patch(SHA2_ELF, vec![]);
    state.add_input_stream(&input);

    let seg_size = SEG_SIZE;
    let seg_path = "/tmp/zkm/sha2";

    let (_total_steps, seg_num, mut state) = split_prog_into_segs(state, seg_path, "", seg_size);

    let start = Instant::now();
    let size = if seg_num == 1 {
        let seg_file = format!("{seg_path}/{}", 0);
        prove_single_seg_common(&seg_file, "", "", "")
    } else {
        prove_multi_seg_common(seg_path, "", "", "", seg_num, 0)
    };
    let end = Instant::now();
    let duration = end.duration_since(start);

    let _hash =  state.read_public_values::<[u8; 32]>();

    (duration, size)
}

fn benchmark_sha3(num_bytes: usize) -> (Duration, usize) {
    let input = vec![5u8; num_bytes];
    let mut state = load_elf_with_patch(SHA3_ELF, vec![]);
    state.add_input_stream(&input);

    let seg_size = SEG_SIZE;
    let seg_path = "/tmp/zkm/sha3";

    let (_total_steps, seg_num, mut state) = split_prog_into_segs(state, seg_path, "", seg_size);

    let start = Instant::now();
    let size = if seg_num == 1 {
        let seg_file = format!("{seg_path}/{}", 0);
        prove_single_seg_common(&seg_file, "", "", "")
    } else {
        prove_multi_seg_common(seg_path, "", "", "", seg_num, 0)
    };
    let end = Instant::now();
    let duration = end.duration_since(start);

    let _hash =  state.read_public_values::<[u8; 32]>();

    (duration, size)
}

fn benchmark_fibonacci(n: u32) -> (Duration, usize) {
    let mut state = load_elf_with_patch(FIBONACCI_ELF, vec![]);
    state.add_input_stream(&n);

    let seg_size = SEG_SIZE;
    let seg_path = "/tmp/zkm/fibonacci";

    let (_total_steps, seg_num, mut state) = split_prog_into_segs(state, seg_path, "", seg_size);

    let start = Instant::now();
    let size = if seg_num == 1 {
        let seg_file = format!("{seg_path}/{}", 0);
        prove_single_seg_common(&seg_file, "", "", "")
    } else {
        prove_multi_seg_common(seg_path, "", "", "", seg_num, 0)
    };
    let end = Instant::now();
    let duration = end.duration_since(start);

    let _output = state.read_public_values::<u128>();
    (duration, size)
}

fn benchmark_bigmem(value: u32) -> (Duration, usize) {
    let mut state = load_elf_with_patch(BIGMEM_ELF, vec![]);
    state.add_input_stream(&value);

    let seg_size = SEG_SIZE;
    let seg_path = "/tmp/zkm/bigmem";

    let (_total_steps, seg_num, mut state) = split_prog_into_segs(state, seg_path, "", seg_size);

    let start = Instant::now();
    let size = if seg_num == 1 {
        let seg_file = format!("{seg_path}/{}", 0);
        prove_single_seg_common(&seg_file, "", "", "")
    } else {
        prove_multi_seg_common(seg_path, "", "", "", seg_num, 0)
    };
    let end = Instant::now();
    let duration = end.duration_since(start);

    let _output = state.read_public_values::<u32>();
    (duration, size)
}
