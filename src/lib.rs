extern crate ndarray;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use ndarray::prelude::*;
use ndarray::Zip;

use itertools::izip;

use std::time::Instant;

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
    fn new() -> Self {
        Gradient {}
    }

    fn jc_current(
        &self,
        _py: Python<'_>,
        phi: Vec<Vec<Vec<Vec<f64>>>>,
        mlt: Vec<Vec<f64>>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        // flatten the received arrays
        let phi = convert(phi);
        let bf_list = mlt[0].len();
        let mlt = mlt.concat();

        let mut jx = Array3::<f64>::zeros((phi.dim().1, phi.dim().2, phi.dim().3));
        let mut jy = Array3::<f64>::zeros((phi.dim().1, phi.dim().2, phi.dim().3));
        let mut jz = Array3::<f64>::zeros((phi.dim().1, phi.dim().2, phi.dim().3));

        let mut grad_x_arr = Vec::with_capacity(bf_list);
        let mut grad_y_arr = Vec::with_capacity(bf_list);
        let mut grad_z_arr = Vec::with_capacity(bf_list);

        for i_orb in 0..bf_list {
            let phi = phi.slice(s![i_orb, .., .., ..]);
            let grad_phi = gradient04(&phi, &[dx, dy, dz]);

            grad_x_arr.push(grad_phi[0].to_owned());
            grad_y_arr.push(grad_phi[1].to_owned());
            grad_z_arr.push(grad_phi[2].to_owned());
        }

        let mut total_idx: usize = 0;


        // NOTE: mlt is a triangular matrix. Right now we're looping through the whole
        // MxN matrix, but it can be sped up by first dividing into diagonal and lower matrix
        // then looping through the lower triangle and doubling its contribution to the current
        // density. The contribution from the diagonal doesn't need to be doubled.
        let now = Instant::now();
        for i_orb in 0..bf_list {
            for (grad_x, grad_y, grad_z) in izip!(&grad_x_arr, &grad_y_arr, &grad_z_arr) {
                let psi = phi.slice(s![i_orb, .., .., ..]);

                Zip::from(&mut jx)
                    .and(&psi)
                    .and(grad_x)
                    .par_apply(|jx, psi, grad_phi| *jx += 2.0 * &mlt[total_idx] * psi * grad_phi);

                Zip::from(&mut jy)
                    .and(&psi)
                    .and(grad_y)
                    .par_apply(|jy, psi, grad_phi| *jy += 2.0 * &mlt[total_idx] * psi * grad_phi);

                Zip::from(&mut jz)
                    .and(&psi)
                    .and(grad_z)
                    .par_apply(|jz, psi, grad_phi| *jz += 2.0 * &mlt[total_idx] * psi * grad_phi);

                total_idx += 1;
            }
        }
        println!("Time for hot loop: {}", now.elapsed().as_millis());
        let d_a: f64 = dx * dy;
        let current = (jz.sum_axis(Axis(0)).sum_axis(Axis(0))) * d_a;

        Ok((
            current.to_vec(),
            jx.into_raw_vec(),
            jy.into_raw_vec(),
            jz.into_raw_vec(),
        ))
    }
}

fn gradient04(f: &ArrayView3<f64>, steps: &[f64; 3]) -> Vec<Array3<f64>> {
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

    for (idx, step) in steps.iter().enumerate() {
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

        out.par_map_inplace(|x| *x /= step);

        result.push(out);
    }
    result
}

#[pyfunction]
fn get_info(_input: Vec<Vec<Vec<Vec<f64>>>>) {}

#[pymodule]
fn libgradient(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_info))?;
    // m.add_wrapped(wrap_pyfunction!(jc_current))?;
    m.add_class::<Gradient>()?;

    Ok(())
}
