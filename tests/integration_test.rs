use std::{
    collections::BTreeMap,
    fs::File,
    io::BufWriter,
    ops::{Add, Div, Mul, Sub},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Mutex,
    time::{Duration, Instant},
};

use num_traits::{NumCast, ToPrimitive, Zero};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use wavelet_rs::{
    decoder::VolumeWaveletDecoder,
    encoder::VolumeWaveletEncoder,
    filter::{Average, AverageFilter, Filter, GenericFilter},
    range::for_each_range,
    stream::{CompressionLevel, Deserializable, Serializable, SerializeStream},
    transformations::{BlockCount, DerivableMetadataFilter, KnownGreedyFilter},
    volume::VolumeBlock,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Fraction {
    frac: fraction::BigFraction,
}

impl Add for Fraction {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac + rhs.frac,
        }
    }
}

impl Sub for Fraction {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac - rhs.frac,
        }
    }
}

impl Mul for Fraction {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac * rhs.frac,
        }
    }
}

impl Div for Fraction {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Fraction {
            frac: self.frac / rhs.frac,
        }
    }
}

impl<T> Add<T> for Fraction
where
    fraction::BigFraction: Add<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac + rhs,
        }
    }
}

impl<T> Sub<T> for Fraction
where
    fraction::BigFraction: Sub<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac - rhs,
        }
    }
}

impl<T> Mul<T> for Fraction
where
    fraction::BigFraction: Mul<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac * rhs,
        }
    }
}

impl<T> Div<T> for Fraction
where
    fraction::BigFraction: Div<T, Output = fraction::BigFraction>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Fraction {
            frac: self.frac / rhs,
        }
    }
}

impl Zero for Fraction {
    fn zero() -> Self {
        0.into()
    }

    fn is_zero(&self) -> bool {
        self.frac.is_zero()
    }
}

impl ToPrimitive for Fraction {
    fn to_i64(&self) -> Option<i64> {
        self.frac.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.frac.to_u64()
    }
}

impl Average for Fraction {
    type Output = Self;

    fn avg(self, rhs: Self) -> Self::Output {
        (self + rhs) / 2
    }
}

impl NumCast for Fraction {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        let n: f64 = n.to_f64()?;
        Some(Fraction { frac: n.into() })
    }
}

impl Serializable for Fraction {
    fn serialize(self, stream: &mut wavelet_rs::stream::SerializeStream) {
        self.frac.to_string().serialize(stream);
    }
}

impl Deserializable for Fraction {
    fn deserialize(stream: &mut wavelet_rs::stream::DeserializeStreamRef<'_>) -> Self {
        let str: String = String::deserialize(stream);

        Fraction {
            frac: fraction::BigFraction::from_str(&str).unwrap(),
        }
    }
}

impl<T> From<T> for Fraction
where
    fraction::BigFraction: From<T>,
{
    fn from(value: T) -> Self {
        Fraction { frac: value.into() }
    }
}

impl From<Fraction> for f64 {
    fn from(value: Fraction) -> Self {
        value.frac.to_f64().unwrap()
    }
}

#[test]
fn test_correctness() {
    let resource_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let resource_path = resource_path.join("resources/integration");

    let block_sizes = [
        vec![1, 1, 1],
        vec![2, 2, 1],
        vec![4, 4, 1],
        vec![8, 8, 1],
        vec![16, 16, 1],
        vec![32, 32, 1],
        vec![64, 64, 1],
    ];

    let mut errors = Vec::new();
    let mut rng = SmallRng::seed_from_u64(123456789);

    errors.extend(test_sample1(&resource_path));
    errors.extend(test_sample2(&resource_path));
    errors.extend(test_sample3(&resource_path));
    errors.extend(test_sample4::<f64>(&resource_path));

    errors.extend(create_and_test::<Fraction>(
        &resource_path.join("random_64x64"),
        "random_64x64",
        &[64, 64, 1],
        &block_sizes,
        &mut rng,
    ));
    errors.extend(create_and_test::<f32>(
        &resource_path.join("random_64x64_f32"),
        "random_64x64_f32",
        &[64, 64, 1],
        &block_sizes,
        &mut rng,
    ));
    errors.extend(create_and_test::<f64>(
        &resource_path.join("random_64x64_f64"),
        "random_64x64_f64",
        &[64, 64, 1],
        &block_sizes,
        &mut rng,
    ));

    errors.extend(create_and_test::<Fraction>(
        &resource_path.join("random_103x96"),
        "random_103x96",
        &[103, 96, 1],
        &block_sizes,
        &mut rng,
    ));
    errors.extend(create_and_test::<f32>(
        &resource_path.join("random_103x96_f32"),
        "random_103x96_f32",
        &[103, 96, 1],
        &block_sizes,
        &mut rng,
    ));
    errors.extend(create_and_test::<f64>(
        &resource_path.join("random_103x96_f64"),
        "random_103x96_f64",
        &[103, 96, 1],
        &block_sizes,
        &mut rng,
    ));

    let error_file_path = resource_path.join("errors.txt");
    let mut error_file = init_error_file(&error_file_path);
    for (name, block_string, errors, greedy_errors) in errors {
        write_error(&mut error_file, &name, &block_string, false, &errors);
        write_error(&mut error_file, &name, &block_string, true, &greedy_errors);
    }
}

fn test_sample1(
    path: &Path,
) -> Vec<(
    String,
    String,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
)> {
    let path = path.join("sample1");
    let volume = VolumeBlock::<Fraction>::new_with_data(
        &[8, 8, 1],
        vec![
            1.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
            8.into(),
            3.into(),
            18.into(),
            15.into(),
            9.into(),
            21.into(),
            12.into(),
            24.into(),
            6.into(),
            7.into(),
            21.into(),
            28.into(),
            14.into(),
            35.into(),
            56.into(),
            49.into(),
            42.into(),
            6.into(),
            12.into(),
            24.into(),
            18.into(),
            48.into(),
            30.into(),
            42.into(),
            36.into(),
            2.into(),
            14.into(),
            10.into(),
            16.into(),
            8.into(),
            12.into(),
            6.into(),
            4.into(),
            4.into(),
            20.into(),
            24.into(),
            8.into(),
            28.into(),
            12.into(),
            16.into(),
            32.into(),
            8.into(),
            56.into(),
            40.into(),
            48.into(),
            32.into(),
            64.into(),
            24.into(),
            16.into(),
            5.into(),
            15.into(),
            10.into(),
            40.into(),
            20.into(),
            25.into(),
            35.into(),
            30.into(),
        ],
    )
    .unwrap();

    let blocks = [vec![1, 1, 1], vec![2, 2, 1], vec![4, 4, 1], vec![8, 8, 1]];
    test_volume(&path, "sample1", &[8, 8, 1], &blocks, volume)
}

fn test_sample2(
    path: &Path,
) -> Vec<(
    String,
    String,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
)> {
    let path = path.join("sample2");
    let volume = VolumeBlock::<Fraction>::new_with_data(
        &[9, 8, 1],
        vec![
            1.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
            8.into(),
            9.into(),
            3.into(),
            27.into(),
            15.into(),
            9.into(),
            21.into(),
            12.into(),
            24.into(),
            6.into(),
            18.into(),
            7.into(),
            21.into(),
            28.into(),
            14.into(),
            35.into(),
            56.into(),
            49.into(),
            42.into(),
            63.into(),
            6.into(),
            12.into(),
            24.into(),
            18.into(),
            48.into(),
            30.into(),
            42.into(),
            36.into(),
            54.into(),
            2.into(),
            14.into(),
            10.into(),
            16.into(),
            8.into(),
            12.into(),
            6.into(),
            4.into(),
            18.into(),
            4.into(),
            20.into(),
            24.into(),
            8.into(),
            28.into(),
            12.into(),
            16.into(),
            32.into(),
            36.into(),
            8.into(),
            56.into(),
            40.into(),
            48.into(),
            32.into(),
            64.into(),
            24.into(),
            16.into(),
            72.into(),
            5.into(),
            15.into(),
            10.into(),
            40.into(),
            20.into(),
            25.into(),
            35.into(),
            30.into(),
            45.into(),
        ],
    )
    .unwrap();

    let blocks = [vec![1, 1, 1], vec![2, 2, 1], vec![4, 4, 1], vec![8, 8, 1]];
    test_volume(&path, "sample2", &[9, 8, 1], &blocks, volume)
}

fn test_sample3(
    path: &Path,
) -> Vec<(
    String,
    String,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
)> {
    let path = path.join("sample3");
    let volume = VolumeBlock::<Fraction>::new_with_data(
        &[6, 1, 1],
        vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()],
    )
    .unwrap();

    let blocks = [vec![4, 4, 1]];
    test_volume(&path, "sample3", &[6, 1, 1], &blocks, volume)
}

fn test_sample4<T>(
    path: &Path,
) -> Vec<(
    String,
    String,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
)>
where
    T: ErrorComputable + Zero + NumCast + Serializable + Deserializable + Clone + Send + Sync,
    T: Add<Output = T> + Div<Output = T> + Mul<Output = T> + FromUsize + From<i32>,
    GenericFilter<T>: Filter<T>,
    KnownGreedyFilter: DerivableMetadataFilter<BlockCount, T>,
{
    let path = path.join("sample4");
    let volume = VolumeBlock::<T>::new_with_data(
        &[4, 24, 1],
        vec![
            1.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
            8.into(),
            2.into(),
            1.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
            8.into(),
            2.into(),
            3.into(),
            1.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
            8.into(),
            2.into(),
            3.into(),
            4.into(),
            1.into(),
            5.into(),
            6.into(),
            7.into(),
            8.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            1.into(),
            6.into(),
            7.into(),
            8.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            1.into(),
            7.into(),
            8.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
            1.into(),
            8.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
            8.into(),
            1.into(),
            1.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            6.into(),
            8.into(),
            7.into(),
            1.into(),
            2.into(),
            3.into(),
            4.into(),
            5.into(),
            8.into(),
            6.into(),
            7.into(),
            1.into(),
            2.into(),
            3.into(),
            4.into(),
            8.into(),
            5.into(),
            6.into(),
            7.into(),
            1.into(),
            2.into(),
            3.into(),
            8.into(),
            4.into(),
            5.into(),
            6.into(),
            7.into(),
        ],
    )
    .unwrap();

    let blocks = [vec![4, 4, 1]];
    test_volume(&path, "sample4", &[4, 24, 1], &blocks, volume)
}

fn create_and_test<T>(
    path: &Path,
    name: &str,
    dims: &[usize],
    blocks: &[Vec<usize>],
    rng: &mut SmallRng,
) -> Vec<(
    String,
    String,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
)>
where
    T: ErrorComputable + Zero + NumCast + Serializable + Deserializable + Clone + Send + Sync,
    T: Add<Output = T> + Div<Output = T> + Mul<Output = T> + FromUsize + From<f32>,
    GenericFilter<T>: Filter<T>,
    KnownGreedyFilter: DerivableMetadataFilter<BlockCount, T>,
{
    let num_elements = dims.iter().product::<usize>();
    let elements = (0..num_elements)
        .map(|_| {
            let num: f32 = rng.gen();
            <T as From<f32>>::from(num)
        })
        .collect();
    let volume = VolumeBlock::new_with_data(dims, elements).unwrap();

    test_volume(path, name, dims, blocks, volume)
}

fn test_volume<T>(
    path: &Path,
    name: &str,
    dims: &[usize],
    blocks: &[Vec<usize>],
    volume: VolumeBlock<T>,
) -> Vec<(
    String,
    String,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
    BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
)>
where
    T: ErrorComputable + Zero + NumCast + Serializable + Deserializable + Clone + Send + Sync,
    T: Add<Output = T> + Div<Output = T> + Mul<Output = T> + FromUsize,
    GenericFilter<T>: Filter<T>,
    KnownGreedyFilter: DerivableMetadataFilter<BlockCount, T>,
{
    let steps = dims
        .iter()
        .map(|&d| d.next_power_of_two().ilog2())
        .collect::<Vec<_>>();

    let mut volume_map = BTreeMap::new();
    let steps_range = steps.iter().map(|&s| 0..(s as usize) + 1);
    for_each_range(steps_range, |steps| {
        let steps = steps.iter().map(|&s| s as u32).collect::<Vec<_>>();
        let average = create_average(&steps, &volume);

        let steps = steps
            .into_iter()
            .zip(volume.dims())
            .map(|(s, d)| d.next_power_of_two().ilog2() - s)
            .collect::<Vec<_>>();
        volume_map.insert(steps, average);
    });

    let mut all_errors = Vec::new();
    for block in blocks {
        let mut block_string = String::new();
        for dim in block {
            if !block_string.is_empty() {
                block_string.push('x');
            }
            block_string.push_str(&format!("{dim}"));
        }

        let enc_group_path = path.join(format!("{block_string}"));
        let enc_path = enc_group_path.join("standard");
        let enc_path_greedy = enc_group_path.join("greedy");
        let errors = test_with_block_size(
            &enc_path,
            block,
            &volume,
            &volume_map,
            false,
            CompressionLevel::Default,
        );
        let greedy_errors = test_with_block_size(
            &enc_path_greedy,
            block,
            &volume,
            &volume_map,
            true,
            CompressionLevel::Default,
        );

        let enc_path = enc_group_path.join("standard_uncomp");
        let enc_path_greedy = enc_group_path.join("greedy_uncomp");
        let _ = test_with_block_size(
            &enc_path,
            block,
            &volume,
            &volume_map,
            false,
            CompressionLevel::Off,
        );
        let _ = test_with_block_size(
            &enc_path_greedy,
            block,
            &volume,
            &volume_map,
            true,
            CompressionLevel::Off,
        );

        let error_file_path = enc_group_path.join("errors.txt");
        let mut error_file = init_error_file(&error_file_path);
        write_error(&mut error_file, name, &block_string, false, &errors);
        write_error(&mut error_file, name, &block_string, true, &greedy_errors);

        all_errors.push((name.to_string(), block_string, errors, greedy_errors));
    }

    let error_file_path = path.join("errors.txt");
    let mut error_file = init_error_file(&error_file_path);
    for (name, block_string, errors, greedy_errors) in &all_errors {
        write_error(&mut error_file, name, block_string, false, errors);
        write_error(&mut error_file, name, block_string, true, greedy_errors);
    }

    let volumes_dir = path.join("volumes");
    if volumes_dir.exists() {
        std::fs::remove_dir_all(&volumes_dir).unwrap();
    }
    std::fs::create_dir_all(&volumes_dir).unwrap();

    let compressed_dir = volumes_dir.join("compressed");
    let uncompressed_dir = volumes_dir.join("uncompressed");

    std::fs::create_dir_all(&compressed_dir).unwrap();
    std::fs::create_dir_all(&uncompressed_dir).unwrap();

    for (steps, volume) in volume_map {
        let volume_path = compressed_dir.join(format!("{steps:?}.bin"));
        let volume_uncomp_path = uncompressed_dir.join(format!("{steps:?}.bin"));
        let file = File::create(volume_path).unwrap();
        let file_uncomp = File::create(volume_uncomp_path).unwrap();

        let mut stream = SerializeStream::new();
        volume.serialize(&mut stream);
        stream
            .write_encode(CompressionLevel::Default, file)
            .unwrap();
        stream
            .write_encode(CompressionLevel::Off, file_uncomp)
            .unwrap();
    }

    all_errors
}

fn test_with_block_size<T>(
    path: &Path,
    block: &[usize],
    volume: &VolumeBlock<T>,
    volume_map: &BTreeMap<Vec<u32>, VolumeBlock<T>>,
    greedy: bool,
    compression: CompressionLevel,
) -> BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>
where
    T: ErrorComputable + Zero + NumCast + Serializable + Deserializable + Clone + Send + Sync,
    GenericFilter<T>: Filter<T>,
    KnownGreedyFilter: DerivableMetadataFilter<BlockCount, T>,
{
    if path.exists() {
        std::fs::remove_dir_all(path).expect("unable to cleanup from previous test");
    }
    std::fs::create_dir_all(path).expect("unable to create path");

    let mut encoder = VolumeWaveletEncoder::new(volume.dims(), volume.dims().len() - 1);
    let fetcher = move |idx: &[usize]| volume[[idx[0], idx[1], 0].as_ref()].clone();
    encoder.add_fetcher(&[0], fetcher);

    encoder.encode(path, block, AverageFilter, greedy, compression);

    let mut error_map = BTreeMap::new();
    let decoder = VolumeWaveletDecoder::new(path.join("output.bin"));
    for (steps, truth) in volume_map.iter() {
        let before = Instant::now();
        let decoded = decode(steps, volume.dims(), &decoder);
        let time_required = Instant::now().duration_since(before);

        let num_elements = truth.dims().iter().product::<usize>();
        let (mut min, mut max, mut avg) = T::init_errors();

        assert_eq!(truth.dims(), decoded.dims());
        for (orig, decoded) in truth.flatten().iter().zip(decoded.flatten()) {
            (min, max, avg) = orig
                .clone()
                .error(decoded.clone(), min, max, avg, num_elements);
        }

        let (min, max, avg) = (min.into(), max.into(), avg.into());
        error_map.insert(steps.clone(), (min, max, avg, time_required));
    }

    error_map
}

fn decode<T>(steps: &[u32], dims: &[usize], decoder: &VolumeWaveletDecoder<T>) -> VolumeBlock<T>
where
    T: Zero + Deserializable + Clone + Send + Sync,
    GenericFilter<T>: Filter<T>,
    KnownGreedyFilter: DerivableMetadataFilter<BlockCount, T>,
{
    let range = dims.iter().map(|&d| 0..d).collect::<Vec<_>>();
    let mut data = VolumeBlock::new_zero(&dims).unwrap();

    struct Once;
    impl Once {
        fn consume(self) {}
    }
    let once = Once;
    let writer = |counts: &[usize], size: &[usize]| {
        once.consume();
        let windows = data.window_mut().divide_into_with_size_mut(counts, size);

        let (windows, _, _) = windows.into_raw_parts();
        let windows: Vec<_> = windows.into_iter().map(|w| Mutex::new(Some(w))).collect();

        move |block_idx: usize| {
            let window = &windows[block_idx];
            let mut window = window.lock().unwrap().take().unwrap();

            move |idx: &[usize], elem| {
                window[idx] = elem;
            }
        }
    };

    decoder.decode(writer, &range, steps);
    dedup_volume(steps, data)
}

fn create_average<T>(steps: &[u32], volume: &VolumeBlock<T>) -> VolumeBlock<T>
where
    T: Zero + Add<Output = T> + Div<Output = T> + Mul<Output = T> + FromUsize + Clone,
{
    if steps.iter().all(|&s| s == 0) {
        return volume.clone();
    }

    let mut data = volume.clone();
    let mut counts = VolumeBlock::new_fill(data.dims(), 1usize).unwrap();

    for (dim, &steps) in steps.iter().enumerate() {
        for _ in 0..steps {
            let mut new_dims: Vec<_> = data.dims().into();
            new_dims[dim] = (new_dims[dim] / 2) + (new_dims[dim] % 2);

            let mut new_data = VolumeBlock::new_zero(&new_dims).unwrap();
            let mut new_counts = VolumeBlock::new_zero(&new_dims).unwrap();

            let data_window = data.window();
            let count_window = counts.window();

            let mut new_data_window = new_data.window_mut();
            let mut new_count_window = new_counts.window_mut();

            let data_lanes = data_window.lanes(dim);
            let count_lanes = count_window.lanes(dim);

            let new_data_lanes = new_data_window.lanes_mut(dim);
            let new_count_lanes = new_count_window.lanes_mut(dim);

            for (((data, count), mut new_data), mut new_count) in data_lanes
                .zip(count_lanes)
                .zip(new_data_lanes)
                .zip(new_count_lanes)
            {
                for (i, (new_data, new_count)) in
                    new_data.iter_mut().zip(new_count.iter_mut()).enumerate()
                {
                    let left_idx = 2 * i;
                    let right_idx = (2 * i) + 1;

                    if right_idx >= data.len() {
                        *new_data = data[left_idx].clone();
                        *new_count = count[left_idx];
                    } else {
                        let left = data[left_idx].clone();
                        let right = data[right_idx].clone();

                        let left_count = count[left_idx];
                        let right_count = count[right_idx];

                        let left = left * T::from_usize(left_count);
                        let right = right * T::from_usize(right_count);
                        let count = left_count + right_count;

                        *new_data = (left + right) / T::from_usize(count);
                        *new_count = count;
                    }
                }
            }

            data = new_data;
            counts = new_counts;
        }
    }

    data
}

fn dedup_volume<T>(steps: &[u32], volume: VolumeBlock<T>) -> VolumeBlock<T>
where
    T: Zero + Clone,
{
    let steps = steps
        .iter()
        .zip(volume.dims())
        .map(|(s, d)| d.next_power_of_two().ilog2() - s)
        .collect::<Vec<_>>();

    if steps.iter().all(|&s| s == 0) {
        return volume.clone();
    }

    let mut data = volume;

    for (dim, &steps) in steps.iter().enumerate() {
        for _ in 0..steps {
            let mut new_dims: Vec<_> = data.dims().into();
            new_dims[dim] = (new_dims[dim] / 2) + (new_dims[dim] % 2);

            let mut new_data = VolumeBlock::new_zero(&new_dims).unwrap();

            let data_window = data.window();
            let mut new_data_window = new_data.window_mut();

            let data_lanes = data_window.lanes(dim);
            let new_data_lanes = new_data_window.lanes_mut(dim);

            for (data, mut new_data) in data_lanes.zip(new_data_lanes) {
                for (i, new_data) in new_data.iter_mut().enumerate() {
                    let left_idx = 2 * i;
                    let right_idx = (2 * i) + 1;

                    if right_idx >= data.len() {
                        *new_data = data[left_idx].clone();
                    } else {
                        let left = data[left_idx].clone();
                        *new_data = left;
                    }
                }
            }

            data = new_data;
        }
    }

    data
}

fn init_error_file(path: &Path) -> BufWriter<File> {
    let file = File::create(path).unwrap();
    let mut file = BufWriter::new(file);

    use std::io::Write;
    writeln!(
        &mut file,
        "name, block size, decode steps, method, min error, max error, avg error, time"
    )
    .unwrap();

    file
}

fn write_error(
    file: &mut BufWriter<File>,
    name: &str,
    block_str: &str,
    greedy: bool,
    errors: &BTreeMap<Vec<u32>, (f64, f64, f64, Duration)>,
) {
    use std::io::Write;
    let type_str = if greedy { "greedy" } else { "standard" };

    for (steps, &(min, max, avg, time)) in errors.iter() {
        let msecs = time.as_secs_f64() * 1000.0;
        writeln!(
            file,
            "{name}, {block_str}, {steps:?}, {type_str}, {min}, {max}, {avg}, {msecs}ms"
        )
        .unwrap();
    }
}

trait FromUsize {
    fn from_usize(num: usize) -> Self;
}

impl FromUsize for f32 {
    fn from_usize(num: usize) -> Self {
        num as Self
    }
}

impl FromUsize for f64 {
    fn from_usize(num: usize) -> Self {
        num as Self
    }
}

impl FromUsize for Fraction {
    fn from_usize(num: usize) -> Self {
        num.into()
    }
}

trait ErrorComputable: Into<f64> {
    fn init_errors() -> (Self, Self, Self);

    fn error(
        self,
        observed: Self,
        min: Self,
        max: Self,
        avg: Self,
        count: usize,
    ) -> (Self, Self, Self);
}

impl ErrorComputable for Fraction {
    fn init_errors() -> (Self, Self, Self) {
        (f64::MAX.into(), f64::MIN.into(), 0.0.into())
    }

    fn error(
        self,
        observed: Self,
        mut min: Self,
        mut max: Self,
        mut avg: Self,
        count: usize,
    ) -> (Self, Self, Self) {
        let expected = self.frac;
        let decoded = observed.frac.clone();
        let error = expected - decoded;
        let error = error.abs();

        let error = Fraction { frac: error };
        let scaled = error.frac.clone() / fraction::BigFraction::from(count);

        min = min.min(error.clone());
        max = max.max(error);
        avg.frac += scaled;

        (min, max, avg)
    }
}

impl ErrorComputable for f32 {
    fn init_errors() -> (Self, Self, Self) {
        (f32::MAX, f32::MIN, 0.0)
    }

    fn error(
        self,
        observed: Self,
        mut min: Self,
        mut max: Self,
        mut avg: Self,
        count: usize,
    ) -> (Self, Self, Self) {
        let error = self - observed;
        let error = error.abs();

        min = min.min(error.clone());
        max = max.max(error);
        avg += error / count as f32;

        (min, max, avg)
    }
}

impl ErrorComputable for f64 {
    fn init_errors() -> (Self, Self, Self) {
        (f64::MAX, f64::MIN, 0.0)
    }

    fn error(
        self,
        observed: Self,
        mut min: Self,
        mut max: Self,
        mut avg: Self,
        count: usize,
    ) -> (Self, Self, Self) {
        let error = self - observed;
        let error = error.abs();

        min = min.min(error.clone());
        max = max.max(error);
        avg += error / count as f64;

        (min, max, avg)
    }
}
