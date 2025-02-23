mod model;
mod data;
mod train;

use textplots::{Chart, Plot, Shape};
use crate::train::train_model;
use crate::data::generate_data;

fn main() {
    train_model();

    let (x_test, y_test) = generate_data(10);
    let mut points = vec![];

    for i in 0..x_test.len() {
        points.push((x_test[i], y_test[i]));
    }

    println!("Model Predictions:");
    Chart::new(100, 30, 0.0, 10.0)
        .lineplot(&Shape::Lines(&points))
        .display();
}
