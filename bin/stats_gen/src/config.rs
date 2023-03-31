use fraction::Zero;
use log::{info, trace};
use num_traits::NumCast;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    io::Write,
    ops::{Add, Div, Mul},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant},
};
use wavelet_rs::{
    decoder::VolumeWaveletDecoder,
    encoder::VolumeWaveletEncoder,
    filter::AverageFilter,
    range::for_each_range,
    stream::{CompressionLevel, Deserializable, DeserializeStream, Serializable, SerializeStream},
    transformations::{BlockCount, DerivableMetadataFilter},
    volume::VolumeBlock,
};

use crate::{fraction::Fraction, OutputFilter};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub struct Config {
    datasets: BTreeMap<String, Dataset>,
}

impl Config {
    pub fn create_stats(
        &self,
        output_path: &Path,
        repetitions: u32,
    ) -> Result<Stats, Box<dyn Error>> {
        info!("Creating statistics");
        info!("Number of datasets: {}", self.datasets.len());

        let mut stats = Stats {
            stats: BTreeMap::new(),
        };

        for (name, dataset) in &self.datasets {
            info!("Creating stats for dataset: {:?}", name);
            let dataset_path = output_path.join(name);
            let data_stats = dataset.create_stats(&dataset_path, repetitions)?;
            stats.stats.insert(name.clone(), data_stats);
        }

        Ok(stats)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
struct Dataset {
    shape: Vec<usize>,
    block_sizes: Vec<Vec<usize>>,
    data_types: BTreeSet<DataType>,
    data: DatasetData,

    #[serde(default)]
    uncompressed: bool,
}

impl Dataset {
    fn create_stats(
        &self,
        output_path: &Path,
        repetitions: u32,
    ) -> Result<DatasetStats, Box<dyn Error>> {
        info!("Dataset shape: {:?}", self.shape);
        info!("Block sizes: {:?}", self.block_sizes);
        info!("Data types: {:?}", self.data_types);
        info!("Save uncompressed: {:?}", self.uncompressed);

        if output_path.exists() {
            trace!("Removing directory: {output_path:?}");
            std::fs::remove_dir_all(output_path)?;
        }

        let mut stats = DatasetStats {
            shape: self.shape.clone(),
            stats: Default::default(),
        };

        for data_type in &self.data_types {
            let data_path = output_path.join(format!("{data_type:?}"));

            info!("Creating stats for data type: {data_type:?}");
            info!("Output path: {data_path:?}");

            let data_stats = match data_type {
                DataType::Exact => {
                    let data = self.data.create::<Fraction>(&self.shape)?;
                    DataStats::new(
                        data,
                        &data_path,
                        &self.block_sizes,
                        self.uncompressed,
                        repetitions,
                    )?
                }
                DataType::Float => {
                    let data = self.data.create::<f32>(&self.shape)?;
                    DataStats::new(
                        data,
                        &data_path,
                        &self.block_sizes,
                        self.uncompressed,
                        repetitions,
                    )?
                }
                DataType::Double => {
                    let data = self.data.create::<f64>(&self.shape)?;
                    DataStats::new(
                        data,
                        &data_path,
                        &self.block_sizes,
                        self.uncompressed,
                        repetitions,
                    )?
                }
            };

            stats.stats.insert(*data_type, data_stats);
        }

        Ok(stats)
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum DataType {
    Exact,
    Float,
    Double,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
enum DatasetData {
    Random { seed: u64 },
    Embedded { data: Vec<f32> },
}

impl DatasetData {
    fn create<T>(&self, shape: &[usize]) -> Result<VolumeBlock<T>, Box<dyn Error>>
    where
        T: From<f32>,
    {
        let data = match self {
            DatasetData::Random { seed } => {
                trace!("Creating random data, seed: {seed}");

                let mut rng = SmallRng::seed_from_u64(*seed);
                let num_elements = shape.iter().product::<usize>();
                (0..num_elements)
                    .map(|_| {
                        let num: f32 = rng.gen();
                        <T as From<f32>>::from(num)
                    })
                    .collect::<Vec<T>>()
            }
            DatasetData::Embedded { data } => data.iter().map(|&x| x.into()).collect::<Vec<T>>(),
        };

        let data = VolumeBlock::new_with_data(shape, data)?;
        Ok(data)
    }
}

pub struct Stats {
    stats: BTreeMap<String, DatasetStats>,
}

struct DatasetStats {
    shape: Vec<usize>,
    stats: BTreeMap<DataType, DataStats>,
}

struct DataStats {
    volume_disk_size: u64,
    aggregates_disk_size: u64,
    #[allow(unused)]
    volume_disk_size_uncompressed: Option<u64>,
    #[allow(unused)]
    aggregates_disk_size_uncompressed: Option<u64>,
    exact: BTreeMap<Vec<usize>, RunStats>,
    clamped: BTreeMap<Vec<usize>, RunStats>,
}

impl DataStats {
    fn new<T>(
        data: VolumeBlock<T>,
        output_path: &Path,
        blocks: &[Vec<usize>],
        uncompressed: bool,
        repetitions: u32,
    ) -> Result<Self, Box<dyn Error>>
    where
        T: ErrorComputable + Zero + NumCast + Serializable + Deserializable + Clone + Send + Sync,
        T: Zero + Add<Output = T> + Div<Output = T> + Mul<Output = T> + FromUsize + Clone,
        AverageFilter: DerivableMetadataFilter<BlockCount, T>,
    {
        info!("Creating intermediates");

        let steps = data
            .dims()
            .iter()
            .map(|&d| d.next_power_of_two().ilog2())
            .collect::<Vec<_>>();

        let volumes_path = output_path.join("volumes");
        let compressed_path = volumes_path.join("compressed");
        let uncompressed_path = volumes_path.join("uncompressed");

        if !volumes_path.exists() {
            trace!("Creating dir: {volumes_path:?}");
            std::fs::create_dir_all(&volumes_path)?;
        }

        if !compressed_path.exists() {
            trace!("Creating dir: {compressed_path:?}");
            std::fs::create_dir(&compressed_path)?;
        }

        if !uncompressed_path.exists() && uncompressed {
            trace!("Creating dir: {uncompressed_path:?}");
            std::fs::create_dir(&compressed_path)?;
        }

        info!("Steps: {steps:?}");
        info!("Compressed path: {compressed_path:?}");
        info!("Uncompressed path: {:?}", uncompressed_path);

        let mut volume_map = BTreeMap::new();
        let steps_range = steps.iter().map(|&s| 0..(s as usize) + 1);
        for_each_range(steps_range, |steps| {
            let steps = steps.iter().map(|&s| s as u32).collect::<Vec<_>>();
            let rev_steps = steps
                .iter()
                .zip(data.dims())
                .map(|(&s, &d)| d.next_power_of_two().ilog2() - s)
                .collect::<Vec<_>>();

            info!("Creating intermediate: {rev_steps:?}");

            let average = create_average(&steps, &data);
            let compressed_path = compressed_path.join(format!("{rev_steps:?}.bin"));
            let uncompressed_path = uncompressed_path.join(format!("{rev_steps:?}.bin"));

            let mut stream = SerializeStream::new();
            average.serialize(&mut stream);

            let f = std::fs::File::create(&compressed_path).unwrap();
            stream.write_encode(CompressionLevel::Default, f).unwrap();

            if uncompressed {
                let f = std::fs::File::create(uncompressed_path).unwrap();
                stream.write_encode(CompressionLevel::Off, f).unwrap();
            }

            volume_map.insert(rev_steps, compressed_path);
        });

        let volume_disk_size =
            std::fs::metadata(compressed_path.join(format!("{steps:?}.bin")))?.len();
        let aggregates_disk_size = dir_size(compressed_path)?;

        let volume_disk_size_uncompressed = if uncompressed {
            Some(std::fs::metadata(uncompressed_path.join(format!("{steps:?}.bin")))?.len())
        } else {
            None
        };

        let aggregates_disk_size_uncompressed = if uncompressed {
            Some(dir_size(uncompressed_path)?)
        } else {
            None
        };

        let mut stats = Self {
            volume_disk_size,
            aggregates_disk_size,
            volume_disk_size_uncompressed,
            aggregates_disk_size_uncompressed,
            exact: BTreeMap::new(),
            clamped: BTreeMap::new(),
        };
        for block in blocks {
            info!("Block size: {block:?}");

            let block_path = output_path.join(format!("{block:?}"));

            let exact = test_with_block_size(
                true,
                block,
                &block_path.join("exact"),
                &data,
                &volume_map,
                CompressionLevel::Default,
                repetitions,
            )?;
            let clamped = test_with_block_size(
                false,
                block,
                &block_path.join("clamped"),
                &data,
                &volume_map,
                CompressionLevel::Default,
                repetitions,
            )?;

            stats.exact.insert(block.clone(), exact);
            stats.clamped.insert(block.clone(), clamped);
        }

        Ok(stats)
    }
}

struct RunStats {
    disk_size: u64,
    decompositions: BTreeMap<Vec<u32>, DecompositionStats>,
}

struct DecompositionStats {
    min: f64,
    max: f64,
    avg: f64,
    time: Duration,
}

impl Stats {
    pub fn write_stats(&self, output: &Path, filter: OutputFilter) -> Result<(), Box<dyn Error>> {
        let file = std::fs::File::create(output).unwrap();
        let mut file = std::io::BufWriter::new(file);

        writeln!(
            &mut file,
            "name; data type; shape; block size; method; decode steps; min error; max error; avg error; time (ms)"
        )?;

        for (name, dataset) in &self.stats {
            dataset.write_stats(filter, &mut file, name)?;
        }

        Ok(())
    }

    pub fn write_disk_size_stats(
        &self,
        output: &Path,
        filter: OutputFilter,
    ) -> Result<(), Box<dyn Error>> {
        let file = std::fs::File::create(output).unwrap();
        let mut file = std::io::BufWriter::new(file);

        writeln!(
            &mut file,
            "name; data type; block size; method; volume disk size (bytes); aggregates disk size (bytes); encoded disk size (bytes)"
        )?;

        for (name, dataset) in &self.stats {
            dataset.write_disk_size_stats(filter, &mut file, name)?;
        }

        Ok(())
    }
}

impl DatasetStats {
    fn write_stats(
        &self,
        filter: OutputFilter,
        f: &mut impl Write,
        name: &str,
    ) -> Result<(), Box<dyn Error>> {
        for (data_type, x) in &self.stats {
            x.write_stats(filter, f, name, &format!("{data_type:?}"), &self.shape)?;
        }

        Ok(())
    }

    fn write_disk_size_stats(
        &self,
        filter: OutputFilter,
        f: &mut impl Write,
        name: &str,
    ) -> Result<(), Box<dyn Error>> {
        for (data_type, x) in &self.stats {
            x.write_disk_size_stats(filter, f, name, &format!("{data_type:?}"))?;
        }

        Ok(())
    }
}

impl DataStats {
    fn write_stats(
        &self,
        filter: OutputFilter,
        f: &mut impl Write,
        name: &str,
        data_type: &str,
        shape: &[usize],
    ) -> Result<(), Box<dyn Error>> {
        if matches!(filter, OutputFilter::All | OutputFilter::Clamped) {
            for (block_size, x) in &self.clamped {
                x.write_stats(f, name, data_type, shape, block_size, "clamp")?;
            }
        }

        if matches!(filter, OutputFilter::All | OutputFilter::Exact) {
            for (block_size, x) in &self.exact {
                x.write_stats(f, name, data_type, shape, block_size, "exact")?;
            }
        }

        Ok(())
    }

    fn write_disk_size_stats(
        &self,
        filter: OutputFilter,
        f: &mut impl Write,
        name: &str,
        data_type: &str,
    ) -> Result<(), Box<dyn Error>> {
        if matches!(filter, OutputFilter::All | OutputFilter::Clamped) {
            for (block_size, x) in &self.clamped {
                x.write_disk_size_stats(
                    f,
                    name,
                    data_type,
                    block_size,
                    "clamp",
                    self.volume_disk_size,
                    self.aggregates_disk_size,
                )?;
            }
        }

        if matches!(filter, OutputFilter::All | OutputFilter::Exact) {
            for (block_size, x) in &self.exact {
                x.write_disk_size_stats(
                    f,
                    name,
                    data_type,
                    block_size,
                    "exact",
                    self.volume_disk_size,
                    self.aggregates_disk_size,
                )?;
            }
        }

        Ok(())
    }
}

impl RunStats {
    #[allow(clippy::too_many_arguments)]
    fn write_stats(
        &self,
        f: &mut impl Write,
        name: &str,
        data_type: &str,
        shape: &[usize],
        block_size: &[usize],
        method: &str,
    ) -> Result<(), Box<dyn Error>> {
        for (steps, x) in &self.decompositions {
            x.write_stats(f, name, data_type, shape, block_size, method, steps)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_disk_size_stats(
        &self,
        f: &mut impl Write,
        name: &str,
        data_type: &str,
        block_size: &[usize],
        method: &str,
        volume_disk_size: u64,
        aggregates_disk_size: u64,
    ) -> Result<(), Box<dyn Error>> {
        let disk_size = self.disk_size;
        writeln!(
            f,
            r#"{name:?}; {data_type:?}; "{block_size:?}"; {method:?}; {volume_disk_size}; {aggregates_disk_size}; {disk_size}"#
        )?;

        Ok(())
    }
}

impl DecompositionStats {
    #[allow(clippy::too_many_arguments)]
    fn write_stats(
        &self,
        f: &mut impl Write,
        name: &str,
        data_type: &str,
        shape: &[usize],
        block_size: &[usize],
        method: &str,
        steps: &[u32],
    ) -> Result<(), Box<dyn Error>> {
        let Self {
            min,
            max,
            avg,
            time,
        } = self;
        let msecs = time.as_secs_f64() * 1000.0;
        writeln!(
            f,
            r#"{name:?}; {data_type:?}; "{shape:?}"; "{block_size:?}"; {method:?}; "{steps:?}"; {min}; {max}; {avg}; {msecs}"#
        )?;

        Ok(())
    }
}

pub fn load_config(path: &Path) -> Result<Config, Box<dyn Error>> {
    info!("Loading config from path {path:?}");

    let config = std::fs::read_to_string(path)?;
    trace!("Loaded config {path:?}");

    let config: Config = toml::from_str(&config)?;
    trace!("Parsed config {path:?}");

    Ok(config)
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

        min = min.min(error);
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

        min = min.min(error);
        max = max.max(error);
        avg += error / count as f64;

        (min, max, avg)
    }
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
        let decoded = observed.frac;
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

fn test_with_block_size<T>(
    exact: bool,
    block: &[usize],
    output_path: &Path,
    data: &VolumeBlock<T>,
    volume_map: &BTreeMap<Vec<u32>, PathBuf>,
    compression: CompressionLevel,
    repetitions: u32,
) -> Result<RunStats, Box<dyn Error>>
where
    T: ErrorComputable + Zero + NumCast + Serializable + Deserializable + Clone + Send + Sync,
    AverageFilter: DerivableMetadataFilter<BlockCount, T>,
{
    info!("Creating stats, exact: {exact:?}");
    trace!("Creating directory: {output_path:?}");
    std::fs::create_dir_all(output_path)?;

    trace!("Encoding dataset");
    let mut encoder = VolumeWaveletEncoder::new(data.dims(), data.dims().len() - 1);
    let fetcher = move |idx: &[usize]| data[[idx[0], idx[1], 0].as_ref()].clone();
    encoder.add_fetcher(&[0], fetcher);

    encoder.encode(output_path, block, AverageFilter, exact, compression);
    let encoded_size = dir_size(output_path)?;

    trace!("Decoding dataset");
    let mut stats = RunStats {
        disk_size: encoded_size,
        decompositions: BTreeMap::new(),
    };
    let decoder = VolumeWaveletDecoder::new(output_path.join("output.bin"));
    for (steps, volume_path) in volume_map.iter() {
        let before = Instant::now();
        let mut decoded = None;
        for i in 0..repetitions {
            trace!("Steps {steps:?}, repetition {} of {repetitions}", i + 1);
            decoded.replace(decode(steps, data.dims(), &decoder));
        }
        if let Some(decoded) = decoded {
            let time_required = Instant::now().duration_since(before) / repetitions;

            let f = std::fs::File::open(volume_path)?;
            let stream = DeserializeStream::new_decode(f)?;
            let truth = VolumeBlock::<T>::deserialize(&mut stream.stream());

            let num_elements = truth.dims().iter().product::<usize>();
            let (mut min, mut max, mut avg) = T::init_errors();

            assert_eq!(truth.dims(), decoded.dims());
            for (orig, decoded) in truth.flatten().iter().zip(decoded.flatten()) {
                (min, max, avg) = orig
                    .clone()
                    .error(decoded.clone(), min, max, avg, num_elements);
            }

            let (min, max, avg) = (min.into(), max.into(), avg.into());
            stats.decompositions.insert(
                steps.clone(),
                DecompositionStats {
                    min,
                    max,
                    avg,
                    time: time_required,
                },
            );
        }
    }

    Ok(stats)
}

fn decode<T>(
    steps: &[u32],
    dims: &[usize],
    decoder: &VolumeWaveletDecoder<BlockCount, T, AverageFilter>,
) -> VolumeBlock<T>
where
    T: Zero + Deserializable + Clone + Send + Sync,
    AverageFilter: DerivableMetadataFilter<BlockCount, T>,
{
    let range = dims.iter().map(|&d| 0..d).collect::<Vec<_>>();
    let mut data = VolumeBlock::new_zero(dims).unwrap();

    struct Once;
    impl Once {
        #[allow(unused)]
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
        return volume;
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

fn dir_size(path: impl AsRef<Path>) -> std::io::Result<u64> {
    fn dir_size(mut dir: std::fs::ReadDir) -> std::io::Result<u64> {
        dir.try_fold(0, |acc, file| {
            let file = file?;
            let size = match file.metadata()? {
                data if data.is_dir() => dir_size(std::fs::read_dir(file.path())?)?,
                data => data.len(),
            };
            Ok(acc + size)
        })
    }

    dir_size(std::fs::read_dir(path)?)
}
