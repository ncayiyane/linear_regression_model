use rand::Rng;

pub fn generate_data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();

    for _ in 0..n {
        let x = rng.gen_range(0.0..10.0);
        let noise: f32 = rng.gen_range(-0.5..0.5);
        let y = 2.0 * x + 1.0 + noise; // y = 2x + 1 + noise

        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}
