extern crate ndarray_parallel;
extern crate rayon;
#[macro_use]
extern crate ndarray;
extern crate indicatif;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use indicatif::ProgressBar;

use rayon::prelude::*;

use std::thread;

use ndarray::prelude::*;

use ndarray::Zip;
use ndarray_parallel::prelude::*;

// TODO: Følgende bid kode skal køres i Python
/*
funktionen tager originalt Gles som parameter
Mlt = 1j*Gles/(¤*np.pi)
np.save(path + data_basename '"Gles_dV.npy", Mlt)

*/
fn convert(phi: Vec<Vec<Vec<Vec<f64>>>>) -> Array4<f64> {
    let flattened: Vec<f64> = phi.concat().concat().concat();
    let init = Array4::from_shape_vec(
        (phi.len(), phi[0].len(), phi[0][0].len(), phi[0][0][0].len()),
        flattened,
    );
    init.unwrap()
}

#[pyclass]
struct Gradient {}

#[pymethods]
impl Gradient {
    #[new]
    fn new(obj: &PyRawObject) {
        obj.init(Gradient {});
    }

    fn jc_current(
        &self,
        py: Python<'_>,
        phi: Vec<Vec<Vec<Vec<f64>>>>,
        mlt: Vec<Vec<f64>>,
        dx: f64,
        dy: f64,
        dz: f64,
        bf_list: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        // flatten the received arrays
        let phi = convert(phi);
        let mlt = mlt.concat();

        let mut jx = Array3::<f64>::zeros((phi.dim().1, phi.dim().2, phi.dim().3));
        let mut jy = Array3::<f64>::zeros((phi.dim().1, phi.dim().2, phi.dim().3));
        let mut jz = Array3::<f64>::zeros((phi.dim().1, phi.dim().2, phi.dim().3));

        let mut total_idx: usize = 0;

        let pb = ProgressBar::new(bf_list as u64);
        for i_orb in pb.wrap_iter(0..bf_list) {
            for j_orb in 0..bf_list {
                let psi = phi.slice(s![i_orb, .., .., ..]);
                let phi = phi.slice(s![j_orb, .., .., ..]);

                let result = py.allow_threads(move || gradient04(&phi, &[dx, dy, dz]));

                jx = jx + 2.0 * &mlt[total_idx] * &result[0] * &psi;
                jy = jy + 2.0 * &mlt[total_idx] * &result[1] * &psi;
                jz = jz + 2.0 * &mlt[total_idx] * &result[2] * &psi;

                total_idx += 1;
            }
        }
        let dA: f64 = dx * dy;
        let current = (jz.sum_axis(Axis(0)).sum_axis(Axis(0))) * dA;
        let current = current.to_vec();

        Ok((
            current,
            jx.into_raw_vec(),
            jy.into_raw_vec(),
            jz.into_raw_vec(),
        ))
    }
}

fn gradient04(f: &ArrayView3<f64>, step: &[f64; 3]) -> Vec<Array3<f64>> {
    let slice0 = ndarray::Slice::from(2..-2);
    let slice1 = ndarray::Slice::from(..-4);
    let slice2 = ndarray::Slice::from(1..-3);
    let slice3 = ndarray::Slice::from(3..-1);
    let slice4 = ndarray::Slice::from(4..);

    let forward_edge_0 = ndarray::Slice::from(..2);
    let forward_edge_1 = ndarray::Slice::from(1..3);
    let forward_edge_2 = ndarray::Slice::from(..2);

    let backward_edge_0 = ndarray::Slice::from(-2..);
    let backward_edge_1 = ndarray::Slice::from(-2..);
    let backward_edge_2 = ndarray::Slice::from(-3..-1);

    let mut result = Vec::with_capacity(3);

    for idx in 0..3 {
        let mut out = Array3::<f64>::from_elem(f.dim(), 0.0);

        // out[2:-2] = (f[:-4] - 8*f[1:-3] + 8*f[3:-1] - f[4:])/12.0
        let mut central_part_result = out.slice_axis_mut(Axis(idx), slice0);
        let central_part_1 = &f.slice_axis(Axis(idx), slice1);
        let central_part_2 = &f.slice_axis(Axis(idx), slice2);
        let central_part_3 = &f.slice_axis(Axis(idx), slice3);
        let central_part_4 = &f.slice_axis(Axis(idx), slice4);

        Zip::from(&mut central_part_result)
            .and(central_part_1)
            .and(central_part_2)
            .and(central_part_3)
            .and(central_part_4)
            .par_apply(|out, &val1, &val2, &val3, &val4| {
                *out = (val1 - 8.0 * val2 + 8.0 * val3 - val4) / 12.0;
            });

        // 1D equivalent -- out[0:2] = (f[1:3] - f[0:2])
        let mut forward_part_result = out.slice_axis_mut(Axis(idx), forward_edge_0);
        let forward_part_1 = &f.slice_axis(Axis(idx), forward_edge_1);
        let forward_part_2 = &f.slice_axis(Axis(idx), forward_edge_2);

        Zip::from(&mut forward_part_result)
            .and(forward_part_1)
            .and(forward_part_2)
            .apply(|out, &val1, &val2| {
                *out = val1 - val2;
            });

        // // 1D equivalent -- out[-2:] = (f[-2:] - f[-3:-1])
        let mut backward_part_result = out.slice_axis_mut(Axis(idx), backward_edge_0);
        let backward_part_1 = &f.slice_axis(Axis(idx), backward_edge_1);
        let backward_part_2 = &f.slice_axis(Axis(idx), backward_edge_2);

        Zip::from(&mut backward_part_result)
            .and(backward_part_1)
            .and(backward_part_2)
            .apply(|out, &val1, &val2| {
                *out = val1 - val2;
            });

        out.par_map_inplace(|x| *x /= step[idx]);

        result.push(out);
    }
    result
}

#[pyfunction]
fn get_info(s: &str) {
    let vec: Vec<i32> = (0..600000000).collect();
    let mut arr = Array::from_vec(vec);

    arr.par_map_inplace(|x| *x /= 12);
}

#[pymodule]
fn libgradient(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_info))?;
    // m.add_wrapped(wrap_pyfunction!(jc_current))?;
    m.add_class::<Gradient>()?;

    Ok(())
}
